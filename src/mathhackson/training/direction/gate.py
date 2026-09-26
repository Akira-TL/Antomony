"""仅凭既有反馈选择是否接受方向参数更新，训练教师与运行判断隔离。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .adaptation import DirectionCorrection
from .environment import MAX_TURN

TAPS = (1, 2, 3, 4, 4, 8, 12, 16)
FEATURE_COUNT = 34


@dataclass(frozen=True)
class Feedback:
    error: float
    turn: float
    residual: float


class FeedbackHistory:
    def __init__(self) -> None:
        self.frames: deque[Feedback] = deque(maxlen=16)

    def append(self, feedback: Feedback) -> None:
        if not all(math.isfinite(value) for value in (feedback.error, feedback.turn, feedback.residual)):
            raise ValueError("反馈必须有限")
        self.frames.append(feedback)

    def features(self, offset: float, proposed: float) -> np.ndarray:
        values = [offset / DirectionCorrection.limit, (proposed - offset) / .05]
        for tap in TAPS:
            if tap <= len(self.frames):
                frame = self.frames[-tap]
                values.extend((frame.error / math.pi, frame.turn, frame.residual / MAX_TURN, 1.))
            else:
                values.extend((0., 0., 0., 0.))
        return np.clip(np.asarray(values, dtype=np.float32), -5., 5.)


def accept_proposal(model: DirectionCorrection, proposed: float, accept: bool) -> None:
    if not math.isfinite(proposed) or abs(proposed) > model.limit + 1e-7:
        raise ValueError("候选方向偏移越界或非有限")
    if accept:
        with torch.no_grad():
            model.offset.fill_(proposed)


class UpdateGate(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(FEATURE_COUNT, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def probability(self, features: np.ndarray) -> float:
        if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
            raise ValueError("更新判断输入形状错误或非有限")
        with torch.inference_mode():
            value = float(torch.sigmoid(self(torch.as_tensor(features, dtype=torch.float32))))
        if not math.isfinite(value):
            raise ValueError("更新判断输出非有限")
        return value

    def save(self, path: Path, *, update: int) -> None:
        with path.open("xb") as stream:
            np.savez(stream, update=update, version="direction-gate-v1", taps=TAPS,
                     **{key: value.detach().numpy() for key, value in self.state_dict().items()})

    @classmethod
    def load(cls, path: Path) -> UpdateGate:
        model = cls(0)
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != "direction-gate-v1" or tuple(data["taps"]) != TAPS:
                raise ValueError("更新判断检查点格式不兼容")
            weights = {key: torch.from_numpy(data[key].copy()) for key in model.state_dict()}
            if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
                raise ValueError("更新判断权重非有限")
            model.load_state_dict(weights)
        return model.eval().requires_grad_(False)
