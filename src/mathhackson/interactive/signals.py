"""可编辑墙体下的局部信号；几何遮挡不向模型提供路线或方向答案。"""
from __future__ import annotations

import math

import numpy as np

from mathhackson.colony.geometry import Wall
from mathhackson.training.foraging.signals import LocalSignals, SignalSource
from mathhackson.training.foraging.trails import LocalTrailField, TrailProfile


def visible_from(origin: np.ndarray, points: np.ndarray, walls: list[Wall]) -> np.ndarray:
    visible = np.ones(points.shape[:-1], dtype=np.bool_)
    if not points.size:
        return visible
    axes = tuple(range(points.ndim - 1))
    lower = np.minimum(np.min(points, axis=axes), origin)
    upper = np.maximum(np.max(points, axis=axes), origin)
    for wall in walls:
        extent = wall.extent
        if (upper[0] < wall.x - extent[0] or lower[0] > wall.x + extent[0]
                or upper[1] < wall.y - extent[1] or lower[1] > wall.y + extent[1]):
            continue
        c, s = math.cos(wall.angle), math.sin(wall.angle)
        matrix = np.asarray([[c, s], [-s, c]])
        start = (origin - (wall.x, wall.y)) @ matrix.T
        end = (points - (wall.x, wall.y)) @ matrix.T
        delta = end - start
        low, high = np.zeros_like(visible, dtype=float), np.ones_like(visible, dtype=float)
        intersects = np.ones_like(visible)
        # 线段与旋转后的矩形逐轴求交；平行且在边界外的线段不能算命中。
        for axis, half in enumerate((wall.hx, wall.hy)):
            parallel = np.abs(delta[..., axis]) < 1e-12
            intersects &= ~(parallel & (abs(start[axis]) > half))
            a = np.divide(-half - start[axis], delta[..., axis], out=np.full_like(low, -np.inf), where=~parallel)
            b = np.divide(half - start[axis], delta[..., axis], out=np.full_like(low, np.inf), where=~parallel)
            low = np.maximum(low, np.minimum(a, b))
            high = np.minimum(high, np.maximum(a, b))
        visible &= ~(intersects & (low <= high))
    return visible


class BarrierTrailField(LocalTrailField):
    def __init__(self, profile: TrailProfile, *, half: tuple[float, float] = (14., 10.)) -> None:
        super().__init__(profile, half=half)
        self.walls: list[Wall] = []

    def set_walls(self, walls: list[Wall]) -> None:
        super().set_walls(walls)
        self.walls = list(walls)

    def deposit(self, p: np.ndarray, channel: int, amount: float) -> None:
        if not self.walls or self.profile == "additive-cell-v1":
            return super().deposit(p, channel, amount)
        if not self.enabled:
            return
        if p.shape != (2,) or not np.isfinite(p).all() or not math.isfinite(amount) or amount < 0. or channel not in (0, 1):
            raise ValueError("沉积需要有限位置、非负强度与双信息素通道")
        squared = (self.x[None, :] - p[0]) ** 2 + (self.y[:, None] - p[1]) ** 2
        support = (squared <= 1.95 ** 2) & ~self.blocked
        ys, xs = np.nonzero(support)
        allowed = visible_from(p, np.column_stack((self.x[xs], self.y[ys])), self.walls)
        ys, xs = ys[allowed], xs[allowed]
        patch = min(8., amount) * np.exp(-squared[ys, xs] / (2. * .65 ** 2))
        self.values[channel, ys, xs] = np.maximum(self.values[channel, ys, xs], patch)


class BarrierSignals(LocalSignals):
    def __init__(self, profile: TrailProfile, *, half: tuple[float, float] = (14., 10.)) -> None:
        self.trails = BarrierTrailField(profile, half=half)

    def sample(self, points: np.ndarray, sources: tuple[SignalSource, ...]) -> np.ndarray:
        if not self.trails.walls:
            return super().sample(points, sources)
        receptors = np.zeros((*points.shape[:-1], 8), dtype=np.float32)
        receptors[..., 1:3] = self.trail_samples(points)
        for source in sources:
            response = source.sample(points)
            if not np.any(response):
                continue
            visible = visible_from(np.asarray([source.x, source.y]), points, self.trails.walls)
            receptors += response * visible[..., None]
        # 不允许天线跨过实体墙读取另一侧；points[0] 是当前身体位置。
        receptors *= visible_from(points[0], points, self.trails.walls)[..., None]
        return receptors
