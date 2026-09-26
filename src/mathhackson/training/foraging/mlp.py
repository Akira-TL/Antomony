"""无循环连接的局部多层感知机；固定预处理不增加来源标签。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .policy import Phase

NOVEL_COLUMNS = tuple(i * 8 + j for i in range(9) for j in range(3, 8)) + tuple(range(75, 80))


class FeedforwardPolicy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(86, 14)
            self.middle = nn.Linear(14, 8)
            self.direction = nn.Linear(8, 2)
            self.value = nn.Linear(8, 1)
            with torch.no_grad():
                self.encoder.weight[:, NOVEL_COLUMNS] = 0.
                self.direction.weight.mul_(.1)
                self.direction.bias.copy_(torch.tensor([1., 0.]))
        self.set_phase("signal")

    def set_phase(self, phase: Phase) -> None:
        if phase not in ("signal", "reward", "frozen"):
            raise ValueError("未知训练阶段")
        self.requires_grad_(phase != "frozen")
        if phase == "signal":
            self.value.requires_grad_(False)
        self.train(phase != "frozen")

    def forward(self, observation: torch.Tensor, history: tuple[torch.Tensor, ...] = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.shape[-1] != 78:
            raise ValueError("多层感知机只接收局部信号和六项自身状态")
        receptors = observation[..., :72].reshape(*observation.shape[:-1], 9, 8)
        strength = receptors.mean(dim=-2)
        contrast = (receptors - strength.unsqueeze(-2)) / receptors.amax(dim=-2, keepdim=True).clamp_min(.001)
        inputs = torch.cat((contrast.flatten(-2), strength, observation[..., 72:]), dim=-1)
        hidden = torch.tanh(self.middle(torch.tanh(self.encoder(inputs))))
        raw = self.direction(hidden)
        direction = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
        return direction, hidden, self.value(hidden).squeeze(-1)

    def assert_reserved(self) -> None:
        if bool(self.encoder.weight[:, NOVEL_COLUMNS].any()):
            raise AssertionError("基础课程改变了预留接收器连接")

    def save(self, path: Path, *, update: int, phase: Phase) -> None:
        with path.open("xb") as stream:
            np.savez(stream, version="local-mlp-v1", update=update, phase=phase,
                     **{key: value.detach().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> FeedforwardPolicy:
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != "local-mlp-v1":
                raise ValueError("多层感知机检查点版本不兼容")
            model = cls(0)
            weights = {key: torch.from_numpy(data[key].copy()) for key in model.state_dict()}
            if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
                raise ValueError("权重非有限")
            model.load_state_dict(weights)
        model.set_phase("frozen")
        return model
