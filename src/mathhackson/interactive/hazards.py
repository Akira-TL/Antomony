"""由同一世界时钟驱动的可编辑来源，感知不携带来源类别。"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mathhackson.training.foraging.signals import Response, SignalSource


class Trap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: int = Field(ge=0)
    x: float
    y: float
    born_at: int = Field(ge=0)
    radius: float = Field(default=.5, ge=.2, le=2.)
    injury: float = Field(default=.08, ge=0., le=.25)
    speed_multiplier: float = Field(default=1., ge=.1, le=1.)
    period: int = Field(default=0, ge=0, le=4096)
    motion_amplitude: float = Field(default=0., ge=0., le=2.)
    motion_period: int = Field(default=128, ge=8, le=4096)
    response: Response = (1., 0., 0., 1., 0., 0., 0., 0.)

    @model_validator(mode="after")
    def check_settings(self) -> Trap:
        if self.period and self.period < 4:
            raise ValueError("周期至少四步")
        if any(value < 0. or value > 8. for value in self.response):
            raise ValueError("接收器响应必须在0至8之间")
        return self

    def active(self, tick: int) -> bool:
        return not self.period or (tick - self.born_at) % self.period < self.period // 2

    def position(self, tick: int) -> np.ndarray:
        phase = 2. * math.pi * ((tick - self.born_at) % self.motion_period) / self.motion_period
        return np.asarray([self.x, self.y + self.motion_amplitude * math.sin(phase)], dtype=np.float32)

    def source(self, tick: int) -> SignalSource:
        x, y = self.position(tick)
        return SignalSource(float(x), float(y), self.radius + 1.,
                            2. if self.active(tick) else .5, self.response)
