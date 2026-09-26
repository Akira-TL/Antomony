"""纯方向动作网络。运行输入仅含方向，不接收训练反馈。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

MODEL_VERSION = "direction-motor-v1"
INPUT_NAMES = ("direction_forward", "direction_left")
WIDTH = 16


@dataclass(frozen=True)
class DirectionAction:
    move: bool
    turn: float
    move_probability: float


class DirectionMotor(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(2, WIDTH)
            self.readout = nn.Linear(WIDTH, 2)

    def forward(self, directions: Tensor) -> Tensor:
        return self.readout(torch.tanh(self.encoder(directions)))

    def decide(self, direction: np.ndarray) -> DirectionAction:
        if direction.shape != (2,):
            raise ValueError("动作输入必须且只能包含两个方向分量")
        if not np.isfinite(direction).all():
            raise ValueError("方向分量必须是有限值")
        if not np.isclose(np.linalg.norm(direction), 1., atol=1e-5):
            raise ValueError("动作输入必须是单位方向")
        with torch.inference_mode():
            output = self(torch.as_tensor(direction, dtype=torch.float32))
            probability = float(torch.sigmoid(output[0]))
            turn = float(torch.tanh(output[1]))
        if not np.isfinite([probability, turn]).all():
            raise ValueError("动作输出不是有限值")
        return DirectionAction(probability >= .5, turn, probability)

    def freeze(self) -> DirectionMotor:
        self.requires_grad_(False)
        self.eval()
        return self
