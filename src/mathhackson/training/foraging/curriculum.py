"""局部基础信号的监督方向课程，不以合成正确率代替往返表现。"""
from __future__ import annotations

import math

import numpy as np
import torch

from .signals import receptor_points


def signal_batch(rng: np.random.Generator, size: int, *, budgets: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    points = receptor_points(np.zeros(2, dtype=np.float32), 0.)
    carrying = rng.random(size) < .5
    target = np.where(carrying, 1, rng.choice([0, 2], size))
    if budgets:
        exploration = np.where(rng.random(size) < .5, 0., rng.uniform(.25, 1., size))
        reserve = rng.uniform(.05, 1., size)
        target[exploration == 0.] = 1
    angles = rng.uniform(-math.pi, math.pi, (size, 3))
    axes = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    perpendicular = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    along = np.einsum("pc,btc->bpt", points, axes)
    across = np.einsum("pc,btc->bpt", points, perpendicular)
    amplitude = np.exp(rng.uniform(math.log(.01), math.log(4.), (size, 1, 3)))
    slope = rng.uniform(.15, 1., (size, 1, 3))
    width = rng.uniform(.4, 2., (size, 1, 3))
    basic = amplitude * np.exp(along * slope - .5 * (across / width) ** 2)
    # 只监督存在可辨别线索的方向；无线索探索留给奖励训练，不制造全局目标答案。
    basic[(target == 2), :, 0] = 0.
    receptors = np.zeros((size, 9, 8), dtype=np.float32)
    receptors[:, :, :3] = basic
    own_state = np.column_stack((carrying, rng.random(size) < .05, rng.random(size) < .8,
                                 rng.uniform(-1., 1., size))).astype(np.float32)
    inputs = np.concatenate(((np.log1p(receptors) / math.log(9.)).reshape(size, 72), own_state), axis=1)
    if budgets:
        inputs = np.concatenate((inputs, np.column_stack((exploration, reserve)).astype(np.float32)), axis=1)
    directions = axes[np.arange(size), target].astype(np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(directions)
