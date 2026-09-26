"""仅修改陌生接收器连接；先推理、后反馈、提案、接受或跳过。"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from mathhackson.fast_residual import FastResidualParameter
from mathhackson.training.direction.policy import DirectionMotor
from .environment import LocalObservation
from .policy import DirectionDecision, ForagingPolicy
from .reward import DIRECTIONS, direction_distribution, discounted_returns


class AdaptationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    window: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=.02, gt=0.)
    gamma: float = Field(default=.995, ge=0., le=1.)
    maximum_step_norm: float = Field(default=.05, gt=0.)
    maximum_residual_norm: float = Field(default=1., gt=0.)
    recent_capacity: int = Field(default=8, ge=0)


@dataclass(frozen=True)
class AdaptationProposal:
    number: int
    delta: tuple[float, ...]
    gradient_norm: float
    mean_reward: float
    minimum_reward: float
    novel_response: float
    steps: int


class NovelSignalLearner:
    def __init__(self, policy: ForagingPolicy, motor: DirectionMotor, seed: int,
                 config: AdaptationConfig | None = None) -> None:
        self.config = config or AdaptationConfig()
        self.policy = copy.deepcopy(policy)
        self.policy.set_phase("frozen")
        self.policy.novel_signal.requires_grad_(True)
        self.policy.novel_strength.requires_grad_(True)
        self.motor = copy.deepcopy(motor).freeze()
        self.random = torch.Generator().manual_seed(seed)
        self.parameter = FastResidualParameter(self.weights(), recent_capacity=self.config.recent_capacity)
        self.history: deque[torch.Tensor] = deque(maxlen=16)
        self.log_probabilities: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.responses: list[float] = []
        self.awaiting_feedback = False
        self.proposal: AdaptationProposal | None = None
        self.decisions = self.writes = 0
        self.terminal = False

    def weights(self) -> np.ndarray:
        return torch.cat([p.detach().flatten() for p in self.policy.reserved_parameters()]).numpy().copy()

    def act(self, observation: LocalObservation) -> DirectionDecision:
        if self.awaiting_feedback or self.proposal is not None or self.terminal or len(self.rewards) >= self.config.window:
            raise ValueError("必须先完成反馈和待定修改")
        direction, hidden, value = self.policy(torch.from_numpy(observation.vector()), tuple(self.history))
        distribution = direction_distribution(direction)
        index = torch.multinomial(distribution.probs.detach(), 1, generator=self.random).squeeze(0)
        self.history.append(hidden)
        self.log_probabilities.append(distribution.log_prob(index))
        self.values.append(value.detach())
        self.responses.append(float(observation.receptors[:, 3:].max()))
        self.awaiting_feedback = True
        chosen = DIRECTIONS[index].numpy().copy()
        return DirectionDecision(self.motor.decide(chosen), chosen)

    def feedback(self, reward: float, *, terminal: bool = False) -> None:
        if not self.awaiting_feedback or not math.isfinite(reward):
            raise ValueError("反馈必须对应已经执行的行动且为有限值")
        self.rewards.append(reward)
        self.awaiting_feedback = False
        self.terminal = terminal

    @property
    def ready(self) -> bool:
        return not self.awaiting_feedback and bool(self.rewards) and (self.terminal or len(self.rewards) >= self.config.window)

    def propose(self, next_observation: LocalObservation) -> AdaptationProposal:
        if not self.ready or self.proposal is not None:
            raise ValueError("尚无完整后到反馈，或已有待定提案")
        with torch.no_grad():
            bootstrap = 0. if self.terminal else float(self.policy(torch.from_numpy(next_observation.vector()), tuple(self.history))[2])
        returns = discounted_returns(self.rewards, bootstrap, self.config.gamma)
        advantage = returns - torch.stack(self.values)
        loss = -(torch.stack(self.log_probabilities) * advantage).mean()
        gradients = torch.autograd.grad(loss, self.policy.reserved_parameters())
        gradient = torch.cat([g.flatten() for g in gradients]).detach().numpy()
        if not np.isfinite(gradient).all():
            raise ValueError("非有限梯度不能形成修改")
        delta = -self.config.learning_rate * gradient
        norm = float(np.linalg.norm(delta))
        if norm > self.config.maximum_step_norm:
            delta *= self.config.maximum_step_norm / norm
        self.proposal = AdaptationProposal(self.decisions + 1, tuple(float(x) for x in delta),
                                           float(np.linalg.norm(gradient)), float(np.mean(self.rewards)),
                                           min(self.rewards), max(self.responses), len(self.rewards))
        return self.proposal

    def resolve(self, proposal: AdaptationProposal, *, accept: tuple[bool, bool]) -> bool:
        if proposal is not self.proposal or len(accept) != 2 or any(type(x) is not bool for x in accept):
            raise ValueError("必须明确处理当前个体的当前提案")
        delta = np.asarray(proposal.delta, dtype=np.float32).copy()
        if not np.isfinite(delta).all() or delta.shape != (80,):
            raise ValueError("提案增量无效")
        delta[:40] *= float(accept[0])
        delta[40:] *= float(accept[1])
        # 保留拒绝模块不变，缩小本次增量直到全局残差落在范围内。
        current = self.parameter.fast
        while float(np.linalg.norm(current + delta)) > self.config.maximum_residual_norm and np.any(delta):
            delta *= .5
        changed = bool(np.any(delta) and np.any(self.parameter.effective + delta != self.parameter.effective))
        if changed:
            self.parameter.add_delta(delta)
            with torch.no_grad():
                values = torch.from_numpy(self.parameter.effective)
                self.policy.novel_signal.weight.copy_(values[:40].reshape(8, 5))
                self.policy.novel_strength.weight.copy_(values[40:].reshape(8, 5))
            self.writes += 1
        self.decisions += 1
        self.history = deque((value.detach() for value in self.history), maxlen=16)
        self.log_probabilities.clear()
        self.values.clear()
        self.rewards.clear()
        self.responses.clear()
        self.proposal = None
        return changed

    def restart(self) -> None:
        if self.awaiting_feedback or self.proposal is not None or self.rewards:
            raise ValueError("继承或重启前必须完成终止反馈及修改决定")
        self.history.clear()
        self.terminal = False

    def save(self, path: Path) -> None:
        self.policy.save(path, update=self.writes, phase="frozen")
        with path.with_suffix(".residual.npz").open("xb") as stream:
            np.savez(stream, stable=self.parameter.stable, fast=self.parameter.fast,
                     recent=np.asarray(self.parameter.recent, dtype=np.float32).reshape(-1, 80),
                     decisions=self.decisions, writes=self.writes)
