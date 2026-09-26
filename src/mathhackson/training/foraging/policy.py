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
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.basic_signal = nn.Linear(3, 8)
            self.novel_signal = nn.Linear(5, 8, bias=False)
            self.observation_layer = nn.Linear(76, HIDDEN_WIDTH)
            self.recent_memory = nn.Linear(4 * HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
            self.sparse_memory = nn.Linear(4 * HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False)
            self.direction = nn.Linear(HIDDEN_WIDTH, 2)
            self.value = nn.Linear(HIDDEN_WIDTH, 1)
            with torch.no_grad():
                self.novel_signal.weight.zero_()
                self.recent_memory.weight.zero_()
                self.sparse_memory.weight.zero_()
                self.direction.weight.mul_(.1)
                self.direction.bias.copy_(torch.tensor([1., 0.]))
        self.set_phase("signal")

    def set_phase(self, phase: Phase) -> None:
        if phase not in ("signal", "reward", "frozen"):
            raise ValueError("未知训练阶段")
        self.requires_grad_(phase != "frozen")
        self.novel_signal.requires_grad_(False)
        if phase == "signal":
            self.recent_memory.requires_grad_(False)
            self.sparse_memory.requires_grad_(False)
            self.value.requires_grad_(False)
        self.train(phase != "frozen")

    def forward(self, observation: torch.Tensor, history: tuple[torch.Tensor, ...] = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        receptors = observation[..., :72].reshape(*observation.shape[:-1], 9, 8)
        encoded = torch.tanh(self.basic_signal(receptors[..., :3]) + self.novel_signal(receptors[..., 3:]))
        current = self.observation_layer(torch.cat((encoded.flatten(-2), observation[..., 72:]), dim=-1))
        zero = torch.zeros_like(current)
        recent = torch.cat([history[-lag] if len(history) >= lag else zero for lag in RECENT_LAGS], dim=-1)
        sparse = torch.cat([history[-lag] if len(history) >= lag else zero for lag in SPARSE_LAGS], dim=-1)
        hidden = torch.tanh(current + self.recent_memory(recent) + self.sparse_memory(sparse))
        raw = self.direction(hidden)
        direction = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
        return direction, hidden, self.value(hidden).squeeze(-1)

    def save(self, path: Path, *, update: int, phase: Phase) -> None:
        with path.open("xb") as stream:
            np.savez(stream, version="local-foraging-v1", update=update, phase=phase,
                     recent_lags=RECENT_LAGS, sparse_lags=SPARSE_LAGS,
                     **{key: value.detach().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> ForagingPolicy:
        model = cls(0)
        with np.load(path, allow_pickle=False) as data:
            if (str(data["version"]) != "local-foraging-v1" or tuple(data["recent_lags"]) != RECENT_LAGS
                    or tuple(data["sparse_lags"]) != SPARSE_LAGS):
                raise ValueError("局部往返检查点版本不兼容")
            weights = {key: torch.from_numpy(data[key].copy()) for key in model.state_dict()}
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
