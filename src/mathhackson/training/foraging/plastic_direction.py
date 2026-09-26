"""隔离的可塑方向连接与反馈调制；不包含训练循环或已有模型接入。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.distributions import Bernoulli

from mathhackson.training.direction.policy import DirectionAction, DirectionMotor
from .policy import RECENT_LAGS, SPARSE_LAGS
from .reward import DIRECTIONS, direction_distribution


@dataclass(frozen=True)
class PendingDirection:
    features: torch.Tensor
    chosen: torch.Tensor
    expected: torch.Tensor


@dataclass(frozen=True)
class PlasticState:
    fast: torch.Tensor
    eligibility: torch.Tensor
    history: tuple[torch.Tensor, ...]
    previous_reward: torch.Tensor
    rewards: tuple[torch.Tensor, ...]
    last_write_norm: torch.Tensor
    pending: PendingDirection | None = None


@dataclass(frozen=True)
class DirectionStep:
    directions: torch.Tensor
    probabilities: torch.Tensor
    log_probability: torch.Tensor
    actions: tuple[DirectionAction, ...]
    state: PlasticState


@dataclass(frozen=True)
class FeedbackStep:
    state: PlasticState
    boundary: bool
    accepted: torch.Tensor
    modulation: torch.Tensor
    gate_probability: torch.Tensor
    gate_logp: torch.Tensor | None
    write_norm: torch.Tensor
    statistics: torch.Tensor


def _tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
            or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name}须为匹配形状的有限CPU float32张量")


def _limit(value: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value.flatten(1), dim=1).reshape(-1, 1, 1)
    return value * (maximum / norm.clamp_min(maximum))


class PlasticDirection(nn.Module):
    def __init__(self, motor: DirectionMotor, *, seed: int = 0, feature_width: int = 45,
                 max_step: float = .05, max_fast: float = .5) -> None:
        super().__init__()
        if (type(feature_width) is not int or feature_width < 1 or not math.isfinite(max_step)
                or not math.isfinite(max_fast) or not 0 < max_step <= max_fast):
            raise ValueError("特征宽度和快权重范数上限无效")
        self.feature_width, self.max_step, self.max_fast = feature_width, max_step, max_fast
        self.motor = copy.deepcopy(motor).freeze()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(8, 8)
            self.output = nn.Linear(8, 2)
        self.recent_weights = nn.Parameter(torch.zeros(4, 8))
        self.sparse_weights = nn.Parameter(torch.zeros(4, 8))
        self.decay_logit = nn.Parameter(torch.zeros(()))

    def initial_state(self, batch_size: int = 1) -> PlasticState:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("批次必须包含至少一个个体")
        return PlasticState(torch.zeros(batch_size, self.feature_width, 2),
            torch.zeros(batch_size, self.feature_width, 2), (), torch.zeros(batch_size), (), torch.zeros(batch_size))

    def _validate(self, state: PlasticState) -> int:
        if not isinstance(state, PlasticState) or not isinstance(state.fast, torch.Tensor) or state.fast.ndim != 3:
            raise ValueError("在线状态不匹配")
        batch = state.fast.shape[0]
        if batch < 1 or len(state.history) > 16 or len(state.rewards) > 3:
            raise ValueError("在线状态批次或历史长度错误")
        _tensor(state.fast, (batch, self.feature_width, 2), "快权重")
        _tensor(state.eligibility, (batch, self.feature_width, 2), "资格迹")
        _tensor(state.previous_reward, (batch,), "前步反馈")
        _tensor(state.last_write_norm, (batch,), "最近写入范数")
        if bool((state.last_write_norm < 0).any()) or bool((state.fast.flatten(1).norm(dim=1) > self.max_fast + 1e-6).any()):
            raise ValueError("在线状态范数越界")
        for item in state.history:
            _tensor(item, (batch, 8), "反馈帧历史")
        for item in state.rewards:
            _tensor(item, (batch,), "反馈窗口")
        if state.pending is not None:
            _tensor(state.pending.features, (batch, self.feature_width), "待反馈特征")
            _tensor(state.pending.chosen, (batch, 2), "待反馈方向")
            _tensor(state.pending.expected, (batch, 2), "先验期望方向")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise ValueError("模型参数须为有限CPU float32张量")
        return batch

    @staticmethod
    def _generators(generators: tuple[torch.Generator, ...] | None, batch: int) -> tuple[torch.Generator, ...]:
        if (generators is None or len(generators) != batch or len({id(g) for g in generators}) != batch
                or any(not isinstance(g, torch.Generator) or g.device.type != "cpu" for g in generators)):
            raise ValueError("采样须显式提供各个体独立的CPU随机流")
        return generators

    def infer(self, base_direction: torch.Tensor, features: torch.Tensor, state: PlasticState, *,
              sampled: bool = False, generators: tuple[torch.Generator, ...] | None = None) -> DirectionStep:
        batch = self._validate(state)
        if state.pending is not None:
            raise ValueError("上一次行动尚未反馈")
        _tensor(base_direction, (batch, 2), "基础方向")
        _tensor(features, (batch, self.feature_width), "局部特征")
        if not torch.allclose(base_direction.norm(dim=1), torch.ones(batch), atol=1e-5, rtol=0):
            raise ValueError("基础方向必须是单位方向")
        base, observed = base_direction.detach().clone(), features.detach().clone()
        residual = torch.einsum("bf,bfo->bo", observed, state.fast)
        raw = base + residual
        norm = raw.norm(dim=1, keepdim=True)
        normalized = torch.where(norm > 1e-6, raw / norm.clamp_min(1e-6), base)
        # 零残差逐值保留原方向，同时保留对可塑连接的一阶导数。
        direction = normalized + torch.where((residual == 0).all(dim=1, keepdim=True),
                                             base - normalized, torch.zeros_like(base)).detach()
        _tensor(direction, (batch, 2), "有效方向")
        distribution = direction_distribution(direction)
        if sampled:
            streams = self._generators(generators, batch)
            choice = torch.stack([torch.multinomial(distribution.probs[i].detach(), 1, generator=streams[i])[0]
                                  for i in range(batch)])
        else:
            choice = distribution.probs.argmax(dim=1)
        chosen = DIRECTIONS[choice].clone()
        expected = distribution.probs @ DIRECTIONS
        pending = PendingDirection(observed, chosen, expected)
        return DirectionStep(chosen, distribution.probs, distribution.log_prob(choice),
            tuple(self.motor.decide(item.detach().numpy()) for item in chosen), replace(state, pending=pending))

    def feedback(self, reward: torch.Tensor, state: PlasticState, *, terminal: bool = False,
                 frozen: bool = False, sampled_gate: bool = False,
                 generators: tuple[torch.Generator, ...] | None = None,
                 executed_direction: torch.Tensor | None = None,
                 accepted_override: torch.Tensor | None = None) -> FeedbackStep:
        batch = self._validate(state)
        pending = state.pending
        if pending is None:
            raise ValueError("反馈之前必须先推理并执行行动")
        _tensor(reward, (batch,), "当前反馈")
        boundary = len(state.rewards) + 1 == 4 or terminal
        if accepted_override is not None:
            if not boundary or frozen or sampled_gate:
                raise ValueError("接受覆盖仅用于未冻结的更新边界，且不能同时采样接受")
            if (not isinstance(accepted_override, torch.Tensor) or accepted_override.shape != (batch,)
                    or accepted_override.dtype != torch.bool or accepted_override.device.type != "cpu"):
                raise ValueError("接受覆盖须为逐个体的CPU bool张量")
        chosen = pending.chosen
        if executed_direction is not None:
            _tensor(executed_direction, (batch, 2), "已执行方向")
            if not bool(torch.isclose(executed_direction[:, None, :], DIRECTIONS[None], atol=1e-6, rtol=0).all(dim=2).any(dim=1).all()):
                raise ValueError("已执行方向必须属于现有16方向集合")
            chosen = executed_direction.detach().clone()
        reward = reward.detach().clone()
        decay = torch.sigmoid(self.decay_logit)
        eligibility = decay * state.eligibility + (1. - decay) * torch.einsum("bf,bo->bfo", pending.features, chosen - pending.expected)
        rewards = (*state.rewards, reward)
        window = torch.stack(rewards)
        statistics = torch.stack((reward, state.previous_reward, window.mean(dim=0), window.var(dim=0, unbiased=False),
            pending.features.norm(dim=1), eligibility.flatten(1).norm(dim=1), state.fast.flatten(1).norm(dim=1), state.last_write_norm), dim=1)
        _tensor(statistics, (batch, 8), "调制统计")
        statistics = statistics / (1. + statistics.abs())
        memory = torch.zeros(batch, 8)
        for lags, weights in ((RECENT_LAGS, self.recent_weights), (SPARSE_LAGS, self.sparse_weights)):
            frames = torch.stack([state.history[-lag] if len(state.history) >= lag else torch.zeros(batch, 8) for lag in lags], dim=1)
            memory = memory + (frames * torch.tanh(weights)[None]).sum(dim=1)
        hidden = torch.tanh(self.encoder(statistics) + memory / math.sqrt(8.))
        output = self.output(hidden)
        _tensor(output, (batch, 2), "调制输出")
        modulation, logits = torch.tanh(output[:, 0]), output[:, 1]
        gate = Bernoulli(logits=logits)
        accepted = torch.zeros(batch, dtype=torch.bool)
        fast, gate_logp = state.fast, None
        write_norm = torch.zeros(batch)
        if boundary and not frozen:
            if accepted_override is not None:
                accepted = accepted_override.detach().clone()
            elif sampled_gate:
                streams = self._generators(generators, batch)
                accepted = torch.stack([torch.bernoulli(gate.probs[i].detach(), generator=streams[i]) for i in range(batch)]).bool()
            else:
                accepted = gate.probs >= .5
            if accepted_override is None:
                gate_logp = gate.log_prob(accepted.to(torch.float32))
            delta = _limit(self.max_step * modulation[:, None, None] * eligibility, self.max_step)
            proposal = _limit(state.fast + delta, self.max_fast)
            fast = torch.where(accepted[:, None, None], proposal, state.fast)
            write_norm = (fast - state.fast).flatten(1).norm(dim=1)
        next_state = PlasticState(fast, eligibility, (*state.history, hidden)[-16:], reward,
            () if boundary else rewards, write_norm if boundary else state.last_write_norm)
        self._validate(next_state)
        return FeedbackStep(next_state, boundary, accepted, modulation, gate.probs, gate_logp, write_norm, statistics)
