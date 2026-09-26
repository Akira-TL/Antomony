"""带混合局部气味的接触伤害；类别和全局位置不进入蚂蚁观察。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mathhackson.training.direction.policy import DirectionAction
from .colony import ColonyConfig, ColonyEnvironment, ColonyInteraction
from .signals import Response, SignalSource


class DisturbanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    center_fraction: float = Field(default=.55, gt=0., le=1.)
    signal_radius: float = Field(default=1.5, gt=0.)
    signal_strength: float = Field(default=2., ge=0.)
    response: Response = (1., 0., 0., 1., 0., 0., 0., 0.)
    contact_radius: float = Field(default=.5, gt=0.)
    injury_per_step: float = Field(default=.1, ge=0.)
    injury_limit: float = Field(default=1., gt=0.)
    death_cost: float = Field(default=2., ge=0.)
    active_from: int = Field(default=0, ge=0)
    active_until: int | None = Field(default=None, ge=1)


class DisturbedColony(ColonyEnvironment):
    def __init__(self, seed: int, config: ColonyConfig, disturbance: DisturbanceConfig) -> None:
        super().__init__(seed, config)
        self.disturbance = disturbance
        self.source_position = self.food * disturbance.center_fraction
        self.extra_sources = (SignalSource(float(self.source_position[0]), float(self.source_position[1]),
                                           disturbance.signal_radius, disturbance.signal_strength, disturbance.response),)
        self.injuries = np.zeros(config.ants, dtype=np.float32)
        self.killed = np.zeros(config.ants, dtype=np.bool_)

    def step(self, actions: list[DirectionAction]) -> list[ColonyInteraction]:
        tick = self.steps
        active = [not ant.exhausted for ant in self.ants]
        events = super().step(actions)
        hazard = self.disturbance
        if tick < hazard.active_from or (hazard.active_until is not None and tick >= hazard.active_until):
            return events
        for i, ant in enumerate(self.ants):
            if not active[i] or ant.exhausted or float(np.linalg.norm(ant.position - self.source_position)) >= hazard.contact_radius:
                continue
            injury = hazard.injury_per_step
            self.injuries[i] += injury
            killed = float(self.injuries[i]) >= hazard.injury_limit
            self.killed[i] = killed
            ant.exhausted = killed
            events[i] = replace(events[i], exhausted=killed,
                                reward=events[i].reward - injury - hazard.death_cost * float(killed))
        return events
