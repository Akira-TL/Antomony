"""多蚁同步世界：共享场，不共享个体状态；预算不直接决定方向。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydantic import Field

from mathhackson.colony.geometry import move_discs, unit
from mathhackson.training.direction.environment import MAX_TURN, STEP_DISTANCE, wrap_angle
from mathhackson.training.direction.policy import DirectionAction
from .environment import ForagingConfig, LocalObservation
from .signals import LocalSignals, SignalSource, receptor_points


class ColonyConfig(ForagingConfig):
    ants: int = Field(default=8, ge=1, le=32)
    stock: int = Field(default=16, ge=1)
    exploration_steps: int = Field(default=160, ge=1)
    reserve_steps: int = Field(default=160, ge=1)
    return_reward: float = Field(default=2., ge=0.)
    exhaustion_cost: float = Field(default=2., ge=0.)


@dataclass
class Forager:
    position: np.ndarray
    heading: float
    exploration_left: int
    reserve_left: int
    carrying: bool = False
    contact: bool = False
    previous_move: bool = False
    previous_turn: float = 0.
    leg_distance: float = 0.
    release_distance: float = 0.
    away: bool = False
    exhausted: bool = False
    pickups: int = 0
    deliveries: int = 0
    budget_returns: int = 0


@dataclass(frozen=True)
class ColonyInteraction:
    picked_up: bool = False
    delivered: bool = False
    budget_return: bool = False
    exhausted: bool = False
    exploration_reward: float = 0.
    reward: float = 0.


class ColonyEnvironment:
    def __init__(self, seed: int, config: ColonyConfig | None = None) -> None:
        self.config = config or ColonyConfig()
        rng = np.random.default_rng(seed)
        self.home = np.zeros(2, dtype=np.float32)
        angle = float(rng.uniform(-math.pi, math.pi))
        distance = self.config.food_distance_min + float(rng.uniform(0., self.config.food_distance_span))
        self.food = unit(angle) * distance
        self.signals = LocalSignals(trail_profile=self.config.trail_profile)
        self.extra_sources: tuple[SignalSource, ...] = ()
        self.stock = self.config.stock
        self.steps = 0
        # 原八只布局不变；更大群体围巢排布，避免碰撞分离凭空制造位移。
        radius = .5 if self.config.ants <= 8 else .19 / math.sin(math.pi / self.config.ants)
        self.ants = [Forager(radius * unit(2. * math.pi * i / self.config.ants),
                            float(rng.uniform(-math.pi, math.pi)), self.config.exploration_steps,
                            self.config.reserve_steps) for i in range(self.config.ants)]
        if self.config.trail_profile == "bounded-local-v2":
            for ant in self.ants:
                self.signals.trails.deposit(ant.position, 0, self.config.home_rate)

    def food_sources(self) -> tuple[SignalSource, ...]:
        return (SignalSource(float(self.food[0]), float(self.food[1]), self.config.food_radius,
                             self.config.food_strength, (1., 0., 0., 0., 0., 0., 0., 0.)),) if self.stock else ()

    def sources(self) -> tuple[SignalSource, ...]:
        nest = (SignalSource(float(self.home[0]), float(self.home[1]), self.config.nest_signal_radius,
                             self.config.nest_signal_strength, (0., 1., 0., 0., 0., 0., 0., 0.)),
                ) if self.config.nest_signal_strength else ()
        return self.food_sources() + nest + self.extra_sources

    def take_food(self, index: int) -> bool:
        if self.stock > 0 and float(np.linalg.norm(self.ants[index].position - self.food)) < .4:
            self.stock -= 1
            return True
        return False

    def return_food(self, index: int) -> None:
        self.stock += 1

    def observation(self, index: int) -> LocalObservation:
        ant = self.ants[index]
        local = self.signals.sample(receptor_points(ant.position, ant.heading), self.sources())
        return LocalObservation(local, ant.carrying, ant.contact, ant.previous_move, ant.previous_turn,
                                (ant.exploration_left / self.config.exploration_steps,
                                 ant.reserve_left / self.config.reserve_steps))

    @property
    def done(self) -> bool:
        return (self.steps >= self.config.horizon or all(ant.exhausted for ant in self.ants)
                or sum(ant.deliveries for ant in self.ants) >= self.config.stock)

    def movement_scale(self, index: int) -> float:
        return 1.

    def step(self, actions: list[DirectionAction]) -> list[ColonyInteraction]:
        if self.done:
            raise ValueError("回合已结束")
        if len(actions) != len(self.ants) or any(not math.isfinite(a.turn) or not -1. <= a.turn <= 1. for a in actions):
            raise ValueError("每只蚂蚁须有一个有限且合法的行动")
        active = [i for i, ant in enumerate(self.ants) if not ant.exhausted]
        positions = np.stack([self.ants[i].position for i in active])
        displacement = []
        for i in active:
            ant, action = self.ants[i], actions[i]
            ant.heading = wrap_angle(ant.heading + MAX_TURN * action.turn)
            displacement.append(unit(ant.heading) * (STEP_DISTANCE * self.movement_scale(i) if action.move else 0.))
        moved, contacts = move_discs(positions, np.stack(displacement), .18, [], self.signals.trails.half)
        for row, i in enumerate(active):
            ant, action = self.ants[i], actions[i]
            traveled = float(np.linalg.norm(moved[row] - ant.position))
            ant.position = moved[row]
            ant.contact = bool(contacts[row])
            ant.previous_move, ant.previous_turn = action.move, action.turn
            ant.leg_distance += traveled
            ant.release_distance += traveled
            if float(np.linalg.norm(ant.position - self.home)) > .9:
                ant.away = True
            if ant.exploration_left > 0:
                ant.exploration_left -= 1
            else:
                ant.reserve_left -= 1
        self.signals.trails.tick(.1)
        # 全部个体先读取同一时刻的场，再集中释放，避免数组顺序改变探索奖励。
        local = {i: self.observation(i).receptors for i in active}
        outcomes = [ColonyInteraction() for _ in self.ants]
        # 同步争抢最后一份食物时轮转优先级，不固定偏向编号小的个体。
        order = sorted(active, key=lambda i: (i - self.steps) % len(self.ants))
        for i in order:
            ant = self.ants[i]
            visible = float(local[i][:, [0, 2]].max()) >= self.config.signal_threshold
            explore = (self.config.exploration_reward / (1. + float(local[i][0, 1]))
                       if ant.exploration_left > 0 and not ant.carrying
                       and (self.config.explore_with_food_signal or not visible) else 0.)
            strength = math.exp(-ant.leg_distance / self.config.trail_length_scale)
            if not ant.carrying:
                self.signals.trails.deposit(ant.position, 0, self.config.home_rate * strength)
            elif ant.release_distance >= self.config.food_release_spacing:
                self.signals.trails.deposit(ant.position, 1, self.config.food_rate * strength)
                ant.release_distance = 0.
            at_home = float(np.linalg.norm(ant.position - self.home)) < .65
            delivery = ant.carrying and at_home
            pickup = not ant.carrying and self.take_food(i)
            returned = at_home and ant.away and ant.exploration_left == 0
            if pickup:
                ant.carrying = True
                ant.pickups += 1
            elif delivery:
                ant.carrying = False
                ant.deliveries += 1
            if pickup or delivery:
                ant.leg_distance = ant.release_distance = 0.
            if at_home and (ant.away or delivery):
                ant.exploration_left, ant.reserve_left = self.config.exploration_steps, self.config.reserve_steps
                ant.away = False
                ant.leg_distance = ant.release_distance = 0.
            if returned:
                ant.budget_returns += 1
            exhausted = ant.reserve_left <= 0
            ant.exhausted = exhausted
            reward = explore + float(pickup) + 4. * float(delivery)
            # 携食返巢已获交付奖励，同次到巢不再叠加空载补给奖励。
            reward += self.config.return_reward * float(returned and not delivery)
            reward -= self.config.exhaustion_cost * float(exhausted)
            outcomes[i] = ColonyInteraction(pickup, delivery, returned, exhausted, explore, reward)
        self.steps += 1
        return outcomes
