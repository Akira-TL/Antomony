"""只含一只蚂蚁的可见食物课程，行动完全来自模型。"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from mathhackson.colony.geometry import move_discs, unit

Lesson = Literal["straight", "turn", "random"]


class SingleAntEnvironment:
    horizon = 96

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.lesson: Lesson = "random"
        self.max_turn = 10.
        self.reset()

    def reset(self) -> None:
        self.position = np.zeros(2, np.float32)
        self.heading = 0.
        angle = 0. if self.lesson == "straight" else float(self.rng.uniform(-math.pi, math.pi))
        if self.lesson == "turn":
            angle = float(self.rng.choice([-1, 1])) * math.pi / 3
        radius = self.rng.uniform(2.5, 6.5) if self.lesson == "random" else self.rng.uniform(2.5, 4.)
        self.target = unit(angle) * float(radius)
        self.steps = 0
        self.reward = self.progress = self.turn = 0.
        self.turn_bias = 0.
        self.sensor_bias = 0.
        self.move = self.contact = self.reached = False
        self.distance = float(np.linalg.norm(self.target))

    def observation(self) -> np.ndarray:
        relative = self.relative_food_angle() + self.sensor_bias
        return np.asarray([math.cos(relative), math.sin(relative), self.distance / 8., float(self.move), self.turn,
                           np.clip(self.reward, -1., 1.), float(self.contact), self.progress / .18,
                           0., 0., 0., 0., 0., 0., self.steps / self.horizon, 1.], np.float32)

    def relative_food_angle(self) -> float:
        bearing = math.atan2(float(self.target[1] - self.position[1]), float(self.target[0] - self.position[0]))
        return math.atan2(math.sin(bearing - self.heading), math.cos(bearing - self.heading))

    def step(self, move: bool, turn: float) -> float:
        self.move, self.turn = move, turn
        before = abs(self.relative_food_angle())
        actual_turn = float(np.clip(turn + self.turn_bias, -1., 1.))
        self.heading = math.atan2(math.sin(self.heading + math.radians(self.max_turn) * actual_turn),
                                  math.cos(self.heading + math.radians(self.max_turn) * actual_turn))
        displacement = unit(self.heading) * (.18 if move else 0.)
        positions, contacts = move_discs(self.position[None, :], displacement[None, :], .18, [],
                                        np.asarray([14., 10.], np.float32))
        self.position = positions[0]
        distance = float(np.linalg.norm(self.target - self.position))
        self.progress = self.distance - distance
        self.distance = distance
        self.contact = bool(contacts[0])
        self.reached = distance < .4
        self.steps += 1
        after = abs(self.relative_food_angle())
        self.reward = 2. * self.progress + .5 * (before - after) / math.pi - .01 - .1 * self.contact + 2. * self.reached
        return self.reward

    @property
    def done(self) -> bool:
        return self.reached or self.steps >= self.horizon
