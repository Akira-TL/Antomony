"""复用双信息素场，连续采样后与有限支持的来源响应叠加。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .trails import LocalTrailField, TrailProfile

RECEPTOR_COUNT = 8
Response = tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class SignalSource:
    x: float
    y: float
    radius: float
    strength: float
    response: Response

    def __post_init__(self) -> None:
        if len(self.response) != RECEPTOR_COUNT:
            raise ValueError("每个来源必须指定八个接收器响应")
        if not all(math.isfinite(value) for value in (self.x, self.y, self.radius, self.strength, *self.response)):
            raise ValueError("信号源参数必须有限")
        if self.radius <= 0. or self.strength < 0. or min(self.response) < 0.:
            raise ValueError("信号源范围须为正，强度和响应非负")

    def sample(self, points: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(points - np.asarray([self.x, self.y]), axis=-1)
        strength = self.strength * np.maximum(0., 1. - distance / self.radius) ** 2
        return (strength[..., None] * np.asarray(self.response)).astype(np.float32)


class LocalSignals:
    def __init__(self, *, trail_profile: TrailProfile = "additive-cell-v1",
                 half: tuple[float, float] = (14., 10.)) -> None:
        self.trails = LocalTrailField(trail_profile, half=half)

    def trail_samples(self, points: np.ndarray) -> np.ndarray:
        grid = self.trails
        if not grid.enabled:
            return np.zeros((*points.shape[:-1], 2), dtype=np.float32)
        coordinates = (points + grid.half) / (2. * grid.half) * [grid.width, grid.height] - .5
        x = np.clip(coordinates[..., 0], 0., grid.width - 1.)
        y = np.clip(coordinates[..., 1], 0., grid.height - 1.)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, grid.width - 1), np.minimum(y0 + 1, grid.height - 1)
        fx, fy = x - x0, y - y0
        values = (grid.values[:, y0, x0] * (1. - fx) * (1. - fy)
                  + grid.values[:, y0, x1] * fx * (1. - fy)
                  + grid.values[:, y1, x0] * (1. - fx) * fy
                  + grid.values[:, y1, x1] * fx * fy)
        return np.moveaxis(values, 0, -1).astype(np.float32)

    def sample(self, points: np.ndarray, sources: tuple[SignalSource, ...]) -> np.ndarray:
        trails = self.trail_samples(points)
        receptors = np.zeros((*points.shape[:-1], RECEPTOR_COUNT), dtype=np.float32)
        receptors[..., 1:3] = trails
        for source in sources:
            receptors += source.sample(points)
        return receptors


def receptor_points(position: np.ndarray, heading: float) -> np.ndarray:
    facing = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32)
    left = np.asarray([-facing[1], facing[0]], dtype=np.float32)
    axes = np.stack((facing, left, -facing, -left))
    return np.concatenate((position[None, :], position + .45 * axes, position + .9 * axes)).astype(np.float32)
