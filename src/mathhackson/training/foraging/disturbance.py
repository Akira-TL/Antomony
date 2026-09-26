"""局部混合来源的减速、周期作用与预定移动；观察不含来源身份。"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    period_steps: int | None = Field(default=None, ge=2)
    active_steps: int | None = Field(default=None, ge=1)
    inactive_signal_scale: float = Field(default=1., ge=0., le=1.)
    speed_multiplier: float = Field(default=1., gt=0., le=1.)
    slowdown_radius: float = Field(default=.75, gt=0.)
    motion_amplitude: float = Field(default=0., ge=0.)
    motion_period_steps: int = Field(default=128, ge=4)

    @model_validator(mode="after")
    def validate_schedule(self) -> DisturbanceConfig:
        if self.active_until is not None and self.active_until <= self.active_from:
            raise ValueError("结束步数必须晚于开始步数")
        if (self.period_steps is None) != (self.active_steps is None):
            raise ValueError("周期与每周期作用步数须同时指定")
        if self.period_steps is not None and self.active_steps > self.period_steps:
            raise ValueError("每周期作用步数不能超过周期")
        if any(value < 0. for value in self.response):
            raise ValueError("接收器响应必须非负")
        return self

    def active_at(self, tick: int) -> bool:
        if tick < self.active_from or (self.active_until is not None and tick >= self.active_until):
            return False
        return self.period_steps is None or (tick - self.active_from) % self.period_steps < self.active_steps


class DisturbedColony(ColonyEnvironment):
    def __init__(self, seed: int, config: ColonyConfig, disturbance: DisturbanceConfig) -> None:
        super().__init__(seed, config)
        self.disturbance = disturbance
        self.source_center = self.food * disturbance.center_fraction
        self.motion_axis = np.asarray([-self.food[1], self.food[0]], dtype=np.float32) / np.linalg.norm(self.food)
        if np.any(np.abs(self.source_center) + np.abs(self.motion_axis) * disturbance.motion_amplitude
                  > self.signals.trails.half):
            raise ValueError("来源移动路线不能超出场地")
        self.refresh_source()
        self.injuries = np.zeros(config.ants, dtype=np.float32)
        self.killed = np.zeros(config.ants, dtype=np.bool_)

    def refresh_source(self) -> None:
        hazard = self.disturbance
        phase = 2. * math.pi * (self.steps % hazard.motion_period_steps) / hazard.motion_period_steps
        self.source_position = self.source_center + self.motion_axis * (hazard.motion_amplitude * math.sin(phase))
        scale = 1. if hazard.active_at(self.steps) else hazard.inactive_signal_scale
        self.extra_sources = (SignalSource(float(self.source_position[0]), float(self.source_position[1]),
                                           hazard.signal_radius, hazard.signal_strength * scale, hazard.response),)

    def movement_scale(self, index: int) -> float:
        hazard = self.disturbance
        inside = float(np.linalg.norm(self.ants[index].position - self.source_position)) < hazard.slowdown_radius
        return hazard.speed_multiplier if inside and hazard.active_at(self.steps) else 1.

    def step(self, actions: list[DirectionAction]) -> list[ColonyInteraction]:
        tick = self.steps
        active = [not ant.exhausted for ant in self.ants]
        events = super().step(actions)
        hazard = self.disturbance
        if not hazard.active_at(tick):
            self.refresh_source()
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
        self.refresh_source()
        return events
