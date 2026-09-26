"""零初始化记忆保留既有MLP功能；新记忆权重不共享基础参数。"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mlp import FeedforwardPolicy
from .policy import HIDDEN_WIDTH, RECENT_LAGS, SPARSE_LAGS, Phase


class MemoryPolicy(nn.Module):
    def __init__(self, base: FeedforwardPolicy) -> None:
        super().__init__()
        if base.hidden_width != 14:
            raise ValueError("当前记忆模型只支持原14宽基础模型")
        base.assert_reserved()
        self.base = copy.deepcopy(base)
        self.recent_weights = nn.Parameter(torch.zeros(len(RECENT_LAGS), HIDDEN_WIDTH))
        self.sparse_weights = nn.Parameter(torch.zeros(len(SPARSE_LAGS), HIDDEN_WIDTH))
        self.set_phase("frozen")

    def set_phase(self, phase: Phase) -> None:
        if phase not in ("reward", "frozen"):
            raise ValueError("复制的基础网络保持冻结，只支持记忆奖励训练或冻结")
        training = phase == "reward"
        self.train(training)
        self.base.set_phase("frozen")
        self.recent_weights.requires_grad_(training)
        self.sparse_weights.requires_grad_(training)

    def assert_reserved(self) -> None:
        self.base.assert_reserved()

    def memory_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.recent_weights, self.sparse_weights

    def forward(self, observation: torch.Tensor, history: tuple[torch.Tensor, ...] = ()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.base.encode(observation)
        zero = torch.zeros_like(current)
        contribution = zero
        for lags, weights in ((RECENT_LAGS, self.recent_weights), (SPARSE_LAGS, self.sparse_weights)):
            values = [history[-lag] if len(history) >= lag else zero for lag in lags]
            if any(value.shape != current.shape or not bool(torch.isfinite(value).all()) for value in values):
                raise ValueError("记忆状态形状不符或包含非有限值")
            contribution = contribution + (torch.stack(values, dim=-2) * torch.tanh(weights)).sum(dim=-2)
        hidden = torch.tanh(current + contribution / math.sqrt(len(RECENT_LAGS) + len(SPARSE_LAGS)))
        raw = self.base.direction(hidden)
        direction = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
        return direction, hidden, self.base.value(hidden).squeeze(-1)

    def save(self, path: Path, *, update: int, phase: Phase) -> None:
        with path.open("xb") as stream:
            np.savez(stream, version="local-mlp-memory-v1", update=update, phase=phase,
                     recent_lags=RECENT_LAGS, sparse_lags=SPARSE_LAGS,
                     **{key: value.detach().cpu().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> MemoryPolicy:
        with np.load(path, allow_pickle=False) as data:
            if (str(data["version"]) != "local-mlp-memory-v1" or tuple(data["recent_lags"]) != RECENT_LAGS
                    or tuple(data["sparse_lags"]) != SPARSE_LAGS):
                raise ValueError("记忆模型检查点不兼容")
            model = cls(FeedforwardPolicy(0))
            weights = {}
            for key, parameter in model.state_dict().items():
                value = data[key]
                if value.shape != tuple(parameter.shape) or not np.isfinite(value).all():
                    raise ValueError("记忆模型参数维度错误或非有限")
                weights[key] = torch.from_numpy(value.copy()).to(parameter.dtype)
            model.load_state_dict(weights)
        model.assert_reserved()
        model.set_phase("frozen")
        return model
