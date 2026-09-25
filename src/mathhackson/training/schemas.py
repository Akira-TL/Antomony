from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .environment import Lesson
from .model import Group, Phase
from .recurrent import Phase as RecurrentPhase

Task = Literal["normal", "shift", "mixed"]


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["play", "pause", "step", "phase", "lesson", "freeze", "freeze_all", "speed", "turn", "reset"]
    phase: Phase = "motor"
    lesson: Lesson = "random"
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


class TracePoint(BaseModel):
    sequence: int
    tick: int
    episode: int
    phase: Phase
    source: Literal["initial", "skip", "self", "outer"]
    status: str
    value: float
    delta: float
    changed: int
    total_change: float


class WeightTrace(BaseModel):
    session: str
    row: int
    column: int
    points: list[TracePoint]


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
    trainable: list[list[bool]]
    weights: list[list[float]]
    initial: list[list[float]]
    self_delta: list[list[float]]
    outer_delta: list[list[float]]
    history: list[EpisodeRecord]


class RecurrentCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["play", "pause", "step", "phase", "task", "speed", "reset"]
    phase: RecurrentPhase = "motor"
    task: Task = "normal"
    speed: Literal[1, 4, 16] = 1


class RecurrentParameterGroup(BaseModel):
    id: str
    label: str
    values: list[list[float]]
    changes: list[list[float]]
    trainable: list[list[bool]]


class RecurrentEpisode(BaseModel):
    episode: int
    phase: RecurrentPhase
    task: Task
    perturbation: float
    reached: bool
    steps: int
    reward: float
    checkpoint: str


class RecurrentState(BaseModel):
    session: str
    paused: bool
    error: str
    phase: RecurrentPhase
    task: Task
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
    reward: float
    total_reward: float
    reached: bool
    perturbation: float
    active_perturbation: float
    outer_updates: int
    hidden: list[float]
    groups: list[RecurrentParameterGroup]
    history: list[RecurrentEpisode]
