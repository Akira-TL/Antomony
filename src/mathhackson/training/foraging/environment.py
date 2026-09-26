"""无目标答案输入、固定释放信息素、按实际交互发放奖励。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mathhackson.colony.geometry import move_discs, unit
from mathhackson.training.direction.environment import MAX_TURN, STEP_DISTANCE, wrap_angle

from .signals import LocalSignals, SignalSource, receptor_points


class ForagingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    horizon: int = Field(default=512, ge=1)
    food_radius: float = Field(default=2.25, gt=0.)
    food_strength: float = Field(default=4., gt=0.)
    food_distance_min: float = Field(default=2.5, gt=1.5)
    food_distance_span: float = Field(default=2., ge=0.)
    home_rate: float = Field(default=.12, gt=0.)
    food_rate: float = Field(default=.25, gt=0.)
    trail_length_scale: float = Field(default=4., gt=0.)
    food_release_spacing: float = Field(default=.32, gt=0.)
    exploration_reward: float = Field(default=.005, ge=0.)
    explore_with_food_signal: bool = False
    signal_threshold: float = Field(default=.001, gt=0.)


@dataclass(frozen=True)
class LocalObservation:
    receptors: np.ndarray
    carrying: bool
    contact: bool
    previous_move: bool
    previous_turn: float

    def vector(self) -> np.ndarray:
        # 固定压缩保留绝对浓度，不对各点单独做单位长度归一化。
        compressed = np.log1p(self.receptors) / math.log(9.)
        return np.concatenate((compressed.ravel(), np.asarray([
            self.carrying, self.contact, self.previous_move, self.previous_turn], dtype=np.float32))).astype(np.float32)


@dataclass(frozen=True)
class Interaction:
    picked_up: bool
    delivered: bool
    exploration_reward: float
    reward: float


class ForagingEnvironment:
    def __init__(self, seed: int, config: ForagingConfig | None = None) -> None:
        self.config = config or ForagingConfig()
        rng = np.random.default_rng(seed)
        self.home = np.zeros(2, dtype=np.float32)
        self.position = self.home.copy()
        self.heading = float(rng.uniform(-math.pi, math.pi))
        angle = float(rng.uniform(-math.pi, math.pi))
        distance = self.config.food_distance_min + float(rng.uniform(0., self.config.food_distance_span))
        self.food = unit(angle) * distance
        self.signals = LocalSignals()
        self.extra_sources: tuple[SignalSource, ...] = ()
        self.carrying = self.contact = self.previous_move = False
        self.previous_turn = 0.
        self.steps = self.pickups = self.deliveries = 0
        self.stock = 2
        self.leg_distance = self.release_distance = 0.

    def sources(self) -> tuple[SignalSource, ...]:
        food = (SignalSource(float(self.food[0]), float(self.food[1]), self.config.food_radius,
                             self.config.food_strength, (1., 0., 0., 0., 0., 0., 0., 0.)),) if self.stock else ()
        return food + self.extra_sources

    def observation(self) -> LocalObservation:
        points = receptor_points(self.position, self.heading)
        receptors = self.signals.sample(points, self.sources())
        return LocalObservation(receptors, self.carrying, self.contact, self.previous_move, self.previous_turn)

    @property
    def done(self) -> bool:
        return self.steps >= self.config.horizon or self.deliveries >= 2

    def step(self, move: bool, turn: float) -> Interaction:
        if self.done:
            raise ValueError("回合已结束")
        if not math.isfinite(turn) or not -1. <= turn <= 1.:
            raise ValueError("转向量必须为 -1 至 1 的有限值")
        self.heading = wrap_angle(self.heading + MAX_TURN * turn)
        displacement = unit(self.heading) * (STEP_DISTANCE if move else 0.)
        positions, contacts = move_discs(self.position[None, :], displacement[None, :], .18, [],
                                        self.signals.trails.half)
        traveled = float(np.linalg.norm(positions[0] - self.position))
        self.position = positions[0]
        self.contact = bool(contacts[0])
        self.previous_move, self.previous_turn = move, turn
        self.leg_distance += traveled
        self.release_distance += traveled
        self.signals.trails.tick(.1)
        local = self.observation().receptors
        food_visible = float(local[:, [0, 2]].max()) >= self.config.signal_threshold
        exploration = (self.config.exploration_reward / (1. + float(local[0, 1]))
                       if not self.carrying and (self.config.explore_with_food_signal or not food_visible) else 0.)
        strength = math.exp(-self.leg_distance / self.config.trail_length_scale)
        if not self.carrying:
            self.signals.trails.deposit(self.position, 0, self.config.home_rate * strength)
        elif self.release_distance >= self.config.food_release_spacing:
            self.signals.trails.deposit(self.position, 1, self.config.food_rate * strength)
            self.release_distance = 0.
        pickup = not self.carrying and self.stock > 0 and float(np.linalg.norm(self.food - self.position)) < .4
        delivery = self.carrying and float(np.linalg.norm(self.home - self.position)) < .65
        if pickup:
            self.carrying = True
            self.stock -= 1
            self.pickups += 1
        elif delivery:
            self.carrying = False
            self.deliveries += 1
        if pickup or delivery:
            self.leg_distance = self.release_distance = 0.
        self.steps += 1
        return Interaction(pickup, delivery, exploration, exploration + float(pickup) + 4. * float(delivery))
