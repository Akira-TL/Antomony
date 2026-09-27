"""版本化的训练轨迹场；旧演示的信息素实现不变。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from mathhackson.colony.pheromone import Pheromones

TrailProfile = Literal["additive-cell-v1", "bounded-local-v2"]


class LocalTrailField(Pheromones):
    def __init__(self, profile: TrailProfile, *, half: tuple[float, float] = (14., 10.)) -> None:
        if profile not in ("additive-cell-v1", "bounded-local-v2"):
            raise ValueError("未知训练轨迹场版本")
        super().__init__(half=half)
        self.profile = profile
        self.x = np.linspace(-self.half[0] + self.half[0] / self.width,
                             self.half[0] - self.half[0] / self.width, self.width)
        self.y = np.linspace(-self.half[1] + self.half[1] / self.height,
                             self.half[1] - self.half[1] / self.height, self.height)

    def deposit(self, p: np.ndarray, channel: int, amount: float) -> None:
        if self.profile == "additive-cell-v1":
            return super().deposit(p, channel, amount)
        if not self.enabled:
            return
        if p.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(amount) or amount < 0. or channel not in (0, 1):
            raise ValueError("沉积需要有限位置、非负强度与双信息素通道")
        # 局部有限范围的浓度上限，不因原地等待次数覆盖上游较强标记。
        squared = (self.x[None, :] - p[0]) ** 2 + (self.y[:, None] - p[1]) ** 2
        patch = min(8., amount) * np.exp(-squared / (2. * .65 ** 2))
        patch[(squared > 1.95 ** 2) | self.blocked] = 0.
        np.maximum(self.values[channel], patch, out=self.values[channel])
