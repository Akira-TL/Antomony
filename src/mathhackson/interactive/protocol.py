"""实时对照的类型化控制与只读展示状态。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mathhackson.colony.geometry import Wall
from .hazards import Trap
from .world import Food

GroupKey = Literal["adaptive", "mlp", "rules"]


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = Field(default=20260927, ge=0, lt=2**25)
    ants: int = Field(default=64, ge=1, le=64)
    horizon: int = Field(default=4096, ge=16, le=32768)
    stock: int = Field(default=768, ge=1, le=20000)


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    kind: Literal["wall", "food", "trap", "erase-wall", "erase-trap", "clear-trails"]
    x: float = Field(default=0., ge=-17., le=17.)
    y: float = Field(default=0., ge=-12., le=12.)
    hx: float = Field(default=.25, ge=.2, le=4.)
    hy: float = Field(default=1.5, ge=.2, le=4.)
    angle: float = Field(default=0., ge=-100., le=100.)
    stock: int = Field(default=64, ge=1, le=4096)
    identifier: int = Field(default=0, ge=0)
    radius: float = Field(default=.5, ge=.2, le=2.)
    injury: float = Field(default=.08, ge=0., le=.25)
    speed_multiplier: float = Field(default=1., ge=.1, le=1.)
    period: int = Field(default=0, ge=0, le=4096)
    motion_amplitude: float = Field(default=0., ge=0., le=2.)

    @field_validator("period")
    @classmethod
    def valid_period(cls, value: int) -> int:
        if 0 < value < 4:
            raise ValueError("周期至少四步，0表示始终启用")
        return value


class Preview(BaseModel):
    valid: bool
    message: str
    tick: int


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["pause", "step", "speed", "learning", "checkpoint"]
    enabled: bool = True
    rate: int = Field(default=1, ge=1, le=8)


class AntView(BaseModel):
    id: int
    x: float
    y: float
    heading: float
    carrying: bool
    pending: bool
    injury: float
    exploration: int
    reserve: int
    delivered: int
    deaths: int
    exhaustions: int
    revivals: int
    writes: int
    receptors: list[float]


class Counts(BaseModel):
    delivered: int
    pickups: int
    deaths: int
    exhaustions: int
    revivals: int
    active: int
    injury: float
    decisions: int
    accepted: int
    writes: int
    stock: int


class TrapView(BaseModel):
    spec: Trap
    position: list[float]
    active: bool


class WorldView(BaseModel):
    key: GroupKey
    counts: Counts
    ants: list[AntView]
    walls: list[Wall]
    foods: list[Food]
    traps: list[TrapView]
    field: str


class Notice(BaseModel):
    tick: int
    message: str


class Frame(BaseModel):
    run_id: str
    tick: int
    horizon: int
    seed: int
    paused: bool
    done: bool
    learning: bool
    rate: int
    step_ms: float
    checkpoint_tick: int
    groups: list[WorldView]
    notices: list[Notice]
    error: str | None = None


class ParameterPoint(BaseModel):
    tick: int
    values: list[float]


class ParameterModule(BaseModel):
    key: str
    label: str
    frozen: bool
    points: list[ParameterPoint]


class ParameterView(BaseModel):
    run_id: str
    tick: int
    group: GroupKey
    individual: int
    modules: list[ParameterModule]
