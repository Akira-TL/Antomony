"""只依赖局部接收器与自身状态的规则导航，无神经网络或全局目标。"""
from __future__ import annotations

import math

import numpy as np

from mathhackson.training.direction.environment import MAX_TURN
from mathhackson.training.direction.policy import DirectionAction
from .environment import LocalObservation


def local_gradient(values: np.ndarray) -> np.ndarray:
    near = np.asarray([values[1] - values[3], values[2] - values[4]], dtype=np.float32) / .9
    far = np.asarray([values[5] - values[7], values[6] - values[8]], dtype=np.float32) / 1.8
    return .5 * (near + far)


class LocalRuleController:
    def __init__(self, seed: int) -> None:
        self.random = np.random.default_rng(seed)
        self.wander = 0.
        self.escape_left = 0
        self.escape_sign = 1.

    def act(self, observation: LocalObservation) -> DirectionAction:
        signals = observation.receptors
        if signals.shape != (9, 8) or not np.isfinite(signals).all() or (signals < 0.).any():
            raise ValueError("规则组需要九位置八接收器的非负有限响应")
        returning = observation.carrying or (observation.budgets is not None and observation.budgets[0] <= 0.)
        if returning:
            target = local_gradient(signals[:, 1])
        elif float(signals[:, 0].max()) >= .001:
            target = local_gradient(signals[:, 0])
        elif float(signals[:, 2].max()) >= .001:
            target = local_gradient(signals[:, 2])
        else:
            target = -local_gradient(signals[:, 1])
        if observation.contact and self.escape_left == 0:
            self.escape_left = 12
            self.escape_sign = float(self.random.choice((-1., 1.)))
        if self.escape_left:
            self.escape_left -= 1
            return DirectionAction(False, self.escape_sign, 0.)
        if float(np.linalg.norm(target)) <= .0001:
            self.wander = float(np.clip(.9 * self.wander + self.random.normal(0., .18), -1., 1.))
            return DirectionAction(True, self.wander, 1.)
        angle = math.atan2(float(target[1]), float(target[0]))
        move = abs(angle) <= math.pi / 3.
        return DirectionAction(move, float(np.clip(angle / MAX_TURN, -1., 1.)), float(move))
