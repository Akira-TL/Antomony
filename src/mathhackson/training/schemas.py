from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .environment import Lesson
from .model import Group, Phase


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["play", "pause", "step", "phase", "lesson", "freeze", "freeze_all", "speed", "turn", "reset"]
    phase: Phase = "motor"
    lesson: Lesson = "straight"
    group: Group = "action"
    frozen: bool = True
    speed: Literal[1, 4, 16] = 1
    max_turn: float = Field(default=10., ge=1., le=30.)


class GroupState(BaseModel):
    id: Group
    label: str
    start: int
    end: int
    frozen: bool
    manual: bool
    reason: str


class EpisodeRecord(BaseModel):
    episode: int
    phase: Phase
    lesson: Lesson
    steps: int
    reward: float
    reached: bool
    reason: str
    self_updates: int
    outer_updates: int
    checkpoint: str


class State(BaseModel):
    session: str
    paused: bool
    error: str
    phase: Phase
    lesson: Lesson
    speed: int
    tick: int
    episode: int
    steps: int
    horizon: int
    x: float
    y: float
    heading: float
    target_x: float
    target_y: float
    distance: float
    max_turn: float
    move: bool
    turn: float
    move_probability: float
    write_probability: float
    write_status: str
    reward: float
    total_reward: float
    self_updates: int
    outer_updates: int
    inputs: list[str]
    observation: list[float]
    groups: list[GroupState]
    weights: list[list[float]]
    initial: list[list[float]]
    self_delta: list[list[float]]
    outer_delta: list[list[float]]
    history: list[EpisodeRecord]
