"""信号表示、双时间尺度循环方向层与独立冻结动作层。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from mathhackson.training.direction.policy import DirectionAction, DirectionMotor
from .environment import LocalObservation

RECENT_LAGS = (1, 2, 3, 4)
SPARSE_LAGS = (4, 8, 12, 16)
HIDDEN_WIDTH = 8
Phase = Literal["signal", "reward", "frozen"]


class ForagingPolicy(nn.Module):
    def __init__(self, seed: int, *, relative_signals: bool = True, budget_inputs: bool = False) -> None:
        super().__init__()
        if budget_inputs and not relative_signals:
            raise ValueError("预算输入只用于相对信号模型")
        self.relative_signals = relative_signals
        self.budget_inputs = budget_inputs
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.basic_signal = nn.Linear(3, 8)
            self.novel_signal = nn.Linear(5, 8, bias=False)
            self.observation_layer = nn.Linear(78 if budget_inputs else 76, HIDDEN_WIDTH)
            self.recent_memory = nn.Linear(4 * HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
            self.sparse_memory = nn.Linear(4 * HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
            self.direction = nn.Linear(HIDDEN_WIDTH, 2)
            self.value = nn.Linear(HIDDEN_WIDTH, 1)
            self.state_scale = nn.Linear(3 if budget_inputs else 1, 8)
            self.basic_strength = nn.Linear(3, 8, bias=False)
            self.novel_strength = nn.Linear(5, 8, bias=False)
            with torch.no_grad():
                self.novel_signal.weight.zero_()
                self.recent_memory.weight.zero_()
                self.sparse_memory.weight.zero_()
                self.direction.weight.mul_(.1)
                self.direction.bias.copy_(torch.tensor([1., 0.]))
                self.state_scale.weight.zero_()
                self.state_scale.bias.zero_()
                self.basic_strength.weight.zero_()
                self.novel_strength.weight.zero_()
        self.set_phase("signal")

    def set_phase(self, phase: Phase) -> None:
        if phase not in ("signal", "reward", "frozen"):
            raise ValueError("未知训练阶段")
        self.requires_grad_(phase != "frozen")
        self.novel_signal.requires_grad_(False)
        self.novel_strength.requires_grad_(False)
        if phase == "signal":
            self.recent_memory.requires_grad_(False)
            self.sparse_memory.requires_grad_(False)
            self.value.requires_grad_(False)
        self.train(phase != "frozen")

    def reserved_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.novel_signal.weight, self.novel_strength.weight

    def with_budgets(self) -> ForagingPolicy:
        """显式扩展旧模型，新输入连接置零，不改变旧检查点文件。"""
        upgraded = ForagingPolicy(0, relative_signals=self.relative_signals, budget_inputs=True)
        with torch.no_grad():
            for key, target in upgraded.state_dict().items():
                source = self.state_dict()[key]
                if target.shape == source.shape:
                    target.copy_(source)
                else:
                    target.zero_()
                    target[:, :source.shape[1]].copy_(source)
        upgraded.set_phase("frozen")
        return upgraded

    def forward(self, observation: torch.Tensor, history: tuple[torch.Tensor, ...] = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.budget_inputs and observation.shape[-1] != 78:
            raise ValueError("预算模型需要两项自身预算输入")
        receptors = observation[..., :72].reshape(*observation.shape[:-1], 9, 8)
        if self.relative_signals:
            strength = receptors.mean(dim=-2, keepdim=True)
            contrast = (receptors - strength) / receptors.amax(dim=-2, keepdim=True).clamp_min(.001)
            encoded = torch.tanh(self.basic_signal(contrast[..., :3]) + self.novel_signal(contrast[..., 3:])
                                 + self.basic_strength(strength[..., :3]) + self.novel_strength(strength[..., 3:]))
        else:
            encoded = torch.tanh(self.basic_signal(receptors[..., :3]) + self.novel_signal(receptors[..., 3:]))
        state = (torch.cat((observation[..., 72:73], observation[..., 76:78]), dim=-1)
                 if self.budget_inputs else observation[..., 72:73])
        scale = 2. * torch.sigmoid(self.state_scale(state))
        encoded = encoded * scale.unsqueeze(-2)
        own = observation[..., 72:78 if self.budget_inputs else 76]
        current = self.observation_layer(torch.cat((encoded.flatten(-2), own), dim=-1))
        zero = torch.zeros_like(current)
        recent = torch.cat([history[-lag] if len(history) >= lag else zero for lag in RECENT_LAGS], dim=-1)
        sparse = torch.cat([history[-lag] if len(history) >= lag else zero for lag in SPARSE_LAGS], dim=-1)
        hidden = torch.tanh(current + self.recent_memory(recent) + self.sparse_memory(sparse))
        raw = self.direction(hidden)
        direction = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
        return direction, hidden, self.value(hidden).squeeze(-1)

    def save(self, path: Path, *, update: int, phase: Phase) -> None:
        with path.open("xb") as stream:
            version = "local-foraging-v4" if self.budget_inputs else "local-foraging-v3" if self.relative_signals else "local-foraging-v2"
            np.savez(stream, version=version, update=update, phase=phase,
                     recent_lags=RECENT_LAGS, sparse_lags=SPARSE_LAGS,
                     **{key: value.detach().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> ForagingPolicy:
        with np.load(path, allow_pickle=False) as data:
            version = str(data["version"])
            if (version not in ("local-foraging-v1", "local-foraging-v2", "local-foraging-v3", "local-foraging-v4") or tuple(data["recent_lags"]) != RECENT_LAGS
                    or tuple(data["sparse_lags"]) != SPARSE_LAGS):
                raise ValueError("局部往返检查点版本不兼容")
            # 旧快照的缩放恰为一，恢复不改变旧策略行为。
            model = cls(0, relative_signals=version in ("local-foraging-v3", "local-foraging-v4"),
                        budget_inputs=version == "local-foraging-v4")
            weights = {key: (value if (version == "local-foraging-v1" and key.startswith("state_scale."))
                             or (version in ("local-foraging-v1", "local-foraging-v2") and key.startswith(("basic_strength.", "novel_strength.")))
                             else torch.from_numpy(data[key].copy())) for key, value in model.state_dict().items()}
            if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
                raise ValueError("模型权重非有限")
            model.load_state_dict(weights)
        model.set_phase("frozen")
        return model


@dataclass(frozen=True)
class DirectionDecision:
    action: DirectionAction
    direction: np.ndarray


class ForagingController:
    def __init__(self, policy: ForagingPolicy, motor: DirectionMotor) -> None:
        self.policy = policy
        self.motor = motor.freeze()
        self.history: deque[torch.Tensor] = deque(maxlen=16)

    def decide(self, observation: LocalObservation) -> DirectionDecision:
        with torch.inference_mode():
            direction, hidden, _ = self.policy(torch.from_numpy(observation.vector()), tuple(self.history))
            self.history.append(hidden)
            array = direction.numpy().copy()
        return DirectionDecision(self.motor.decide(array), array)
