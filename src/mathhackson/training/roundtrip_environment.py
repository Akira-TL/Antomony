"""单蚁两趟搬运课程：局部双通道信息素，不提供巢穴方位。"""
from __future__ import annotations

import math

import numpy as np

from mathhackson.colony.geometry import move_discs, unit
from mathhackson.colony.pheromone import Pheromones


class RoundTripEnvironment:
    horizon = 256
    max_turn = 10.
    food_visibility = 1.5

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.home = np.zeros(2, np.float32)
        self.position = self.home.copy()
        self.heading = 0.
        self.food = unit(float(self.rng.uniform(-math.pi, math.pi))) * float(self.rng.uniform(2.5, 5.))
        self.field = Pheromones()
        self.release_distance = np.zeros(2, np.float32)
        self.steps = self.leg_steps = self.delivered = self.pickups = 0
        self.move = self.contact = self.carrying = False
        self.turn = self.reward = self.progress = 0.

    def _local_scent(self, channel: int) -> tuple[float, float, float]:
        strength = min(1., self.field.sample(self.position, channel) / 4.)
        gradient = self.field.gradient(self.position, channel)
        facing = unit(self.heading)
        left = np.asarray([-facing[1], facing[0]], np.float32)
        forward = float(np.clip(gradient @ facing, -1., 1.))
        lateral = float(np.clip(gradient @ left, -1., 1.))
        return strength, forward, lateral

    def observation(self) -> np.ndarray:
        home_strength, home_forward, home_left = self._local_scent(0)
        food_strength, food_forward, food_left = self._local_scent(1)
        facing = unit(self.heading)
        left = np.asarray([-facing[1], facing[0]], np.float32)
        visible_food = not self.carrying and (
            self.delivered == 0 or float(np.linalg.norm(self.food - self.position)) < self.food_visibility)
        if visible_food:
            offset = self.food - self.position
            direction = offset / max(float(np.linalg.norm(offset)), 1e-6)
            target = (float(direction @ facing), float(direction @ left),
                      min(1., float(np.linalg.norm(offset)) / 8.))
        else:
            target = (0., 0., 0.)
        return np.asarray([
            *target, float(self.move), self.turn, 0.,
            float(self.contact), 0.,
            home_strength, home_forward, home_left,
            food_strength, food_forward, food_left,
            self.steps / self.horizon, 1., float(self.carrying),
        ], np.float32)

    def step(self, move: bool, turn: float, release_home: bool, release_food: bool) -> float:
        if self.done:
            raise ValueError("回合已结束")
        self.move, self.turn = move, turn
        target = self.home if self.carrying else self.food
        before = float(np.linalg.norm(target - self.position))
        self.heading = math.atan2(math.sin(self.heading + math.radians(self.max_turn) * float(np.clip(turn, -1., 1.))),
                                  math.cos(self.heading + math.radians(self.max_turn) * float(np.clip(turn, -1., 1.))))
        displacement = unit(self.heading) * (.18 if move else 0.)
        previous_position = self.position.copy()
        positions, contacts = move_discs(self.position[None, :], displacement[None, :], .18, [],
                                        np.asarray([14., 10.], np.float32))
        self.position = positions[0]
        self.release_distance += float(np.linalg.norm(self.position - previous_position))
        self.contact = bool(contacts[0])
        self.field.tick(.1)
        strength = .42 * math.exp(-self.leg_steps / 40.)
        if release_home and self.release_distance[0] >= .32:
            self.field.deposit(self.position, 0, strength)
            self.release_distance[0] = 0.
        if release_food and self.release_distance[1] >= .32:
            self.field.deposit(self.position, 1, strength)
            self.release_distance[1] = 0.
        self.steps += 1
        self.leg_steps += 1
        self.progress = before - float(np.linalg.norm(target - self.position))
        picked_up = not self.carrying and float(np.linalg.norm(self.food - self.position)) < .4
        delivered = self.carrying and float(np.linalg.norm(self.home - self.position)) < 1.2
        if picked_up:
            self.carrying = True
            self.pickups += 1
            self.leg_steps = 0
        elif delivered:
            self.carrying = False
            self.delivered += 1
            self.leg_steps = 0
        self.reward = (.5 * self.progress + .5 * picked_up + 8. * delivered - .03
                       - .008 * (release_home + release_food) - .1 * self.contact)
        return self.reward

    @property
    def done(self) -> bool:
        return self.delivered >= 2 or self.steps >= self.horizon
