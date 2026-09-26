"""只给方向指令的单个体运动环境；没有目标点或到达输入。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_TURN = math.radians(10.)
STEP_DISTANCE = .18


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class DirectionEnvironment:
    heading: float
    desired: float
    x: float = 0.
    y: float = 0.

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.heading, self.desired, self.x, self.y)):
            raise ValueError("运动状态必须是有限值")

    def observation(self) -> np.ndarray:
        relative = self.desired - self.heading
        return np.array([math.cos(relative), math.sin(relative)], dtype=np.float32)

    def step(self, move: bool, turn: float) -> float:
        if not math.isfinite(turn) or not -1. <= turn <= 1.:
            raise ValueError("转向量必须在 -1 到 1 之间")
        self.heading = wrap_angle(self.heading + MAX_TURN * turn)
        distance = STEP_DISTANCE if move else 0.
        self.x += distance * math.cos(self.heading)
        self.y += distance * math.sin(self.heading)
        return distance * math.cos(self.desired - self.heading)
