"""回巢复活仅重置身体；模型和记忆由执行器继续持有。"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from mathhackson.colony.geometry import unit
from mathhackson.training.direction.policy import DirectionAction
from .colony import ColonyConfig, ColonyInteraction, Forager
from .disturbance import DisturbanceConfig, DisturbedColony
from .signals import LocalSignals


class RevivingColony(DisturbedColony):
    def __init__(self, seed: int, config: ColonyConfig, disturbance: DisturbanceConfig) -> None:
        super().__init__(seed, config, disturbance)
        self.slots = [.5 * unit(2. * math.pi * i / 8) for i in range(8)]
        self.pending = np.zeros(config.ants, dtype=np.bool_)
        self.deaths = np.zeros(config.ants, dtype=np.int64)
        self.terminations = np.zeros(config.ants, dtype=np.int64)
        self.revivals = np.zeros(config.ants, dtype=np.int64)
        self.total_injury = np.zeros(config.ants, dtype=np.float64)
        self.waiting: deque[int] = deque()
        # 扩群也从巢内分批出发，不把初始拥挤解释为探索能力。
        self.signals = LocalSignals(trail_profile=config.trail_profile)
        for i, ant in enumerate(self.ants):
            ant.position = self.slots[i % 8].copy()
            if i >= 8:
                ant.exhausted = self.pending[i] = True
                self.waiting.append(i)
            elif config.trail_profile == "bounded-local-v2":
                self.signals.trails.deposit(ant.position, 0, config.home_rate)

    @property
    def done(self) -> bool:
        return self.steps >= self.config.horizon or sum(a.deliveries for a in self.ants) >= self.config.stock

    def release_waiting(self) -> list[int]:
        revived: list[int] = []
        if self.done:
            return revived
        for _ in range(len(self.waiting)):
            index = self.waiting.popleft()
            free = next((p for p in self.slots if all(
                other.exhausted or np.linalg.norm(other.position - p) >= .36
                for other in self.ants)), None)
            if free is None:
                self.waiting.append(index)
                continue
            previous = self.ants[index]
            self.ants[index] = Forager(free.copy(), previous.heading, self.config.exploration_steps,
                self.config.reserve_steps, pickups=previous.pickups, deliveries=previous.deliveries,
                budget_returns=previous.budget_returns)
            self.pending[index] = self.killed[index] = False
            self.injuries[index] = 0.
            if self.terminations[index]:
                self.revivals[index] += 1
                revived.append(index)
            self.signals.trails.deposit(free, 0, self.config.home_rate)
        return revived

    def step(self, actions: list[DirectionAction]) -> list[ColonyInteraction]:
        active = [not ant.exhausted for ant in self.ants]
        before = self.injuries.copy()
        events = super().step(actions)
        self.total_injury += self.injuries - before
        for i, ant in enumerate(self.ants):
            if not active[i] or not ant.exhausted:
                continue
            self.deaths[i] += int(self.killed[i])
            self.terminations[i] += 1
            if ant.carrying:
                self.return_food(i)
                ant.carrying = False
            self.pending[i] = True
            self.waiting.append(i)
        return events
