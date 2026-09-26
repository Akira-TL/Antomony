"""从已发生的方向误差更新前置方向偏移，基础动作参数始终冻结。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .environment import MAX_TURN
from .policy import DirectionMotor


@dataclass(frozen=True)
class DifferentiableAction:
    move: bool
    turn: Tensor


def local_direction_loss(turn: Tensor, observed_error: float) -> Tensor:
    if not math.isfinite(observed_error):
        raise ValueError("实际方向反馈必须有限")
    # 值取自真实后果；仅梯度使用标称转角导数，不读取环境扰动或未来结果。
    local_error = turn.new_tensor(observed_error) + MAX_TURN * (turn.detach() - turn)
    return 1. - torch.cos(local_error)


class DirectionCorrection(nn.Module):
    limit = .35

    def __init__(self, motor: DirectionMotor) -> None:
        super().__init__()
        self.motor = motor.freeze()
        self.offset = nn.Parameter(torch.zeros(1))

    def plastic_parameters(self) -> tuple[nn.Parameter]:
        return (self.offset,)

    def command(self, direction: np.ndarray) -> DifferentiableAction:
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("方向必须是两个有限分量")
        if not np.isclose(np.linalg.norm(direction), 1., atol=1e-5):
            raise ValueError("方向必须是单位向量")
        forward, left = torch.from_numpy(direction.astype(np.float32))
        angle = self.offset[0]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack((forward * cosine - left * sine, forward * sine + left * cosine))
        output = self.motor(rotated)
        return DifferentiableAction(bool(output[0].detach() >= 0.), torch.tanh(output[1]))

    def project(self) -> None:
        with torch.no_grad():
            if not bool(torch.isfinite(self.offset).all()):
                raise ValueError("方向更新非有限")
            self.offset.clamp_(-self.limit, self.limit)
