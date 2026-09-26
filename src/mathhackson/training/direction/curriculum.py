"""有限预算的方向动作模仿课程与闭环工程验收。"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .checkpoint import MotorSnapshot
from .environment import DirectionEnvironment, MAX_TURN, STEP_DISTANCE, wrap_angle
from .policy import DirectionMotor


class MotorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int = Field(default=1200, ge=1)
    batch_size: int = Field(default=256, ge=8)
    learning_rate: float = Field(default=.01, gt=0, le=.1)
    checkpoint_every: int = Field(default=100, ge=1)


class MotorFrame(BaseModel):
    tick: int
    desired: float
    heading: float
    x: float
    y: float
    move: bool
    turn: float
    error_degrees: float
    forward_progress: float


class MotorTrial(BaseModel):
    initial_error_degrees: float
    command_changes: int
    steady_mean_error_degrees: float
    steady_max_error_degrees: float
    steady_progress_fraction: float
    passed: bool
    frames: list[MotorFrame]


class MotorEvaluation(BaseModel):
    trials: list[MotorTrial]
    passed: bool
    max_steady_error_degrees: float
    min_steady_progress_fraction: float


def direction_batch(rng: np.random.Generator, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = rng.uniform(-math.pi, math.pi, size)
    # 均匀方位覆盖掉头，近零加密避免仅学会饱和左右转。
    angles[:size // 2] = rng.uniform(-2. * MAX_TURN, 2. * MAX_TURN, size // 2)
    directions = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    move = (np.abs(angles) < math.pi / 3).astype(np.float32)
    turn = np.clip(angles / MAX_TURN, -1., 1.).astype(np.float32)
    return torch.from_numpy(directions), torch.from_numpy(move), torch.from_numpy(turn)


def train_motor(seed: int, config: MotorConfig,
                save: Callable[[DirectionMotor, MotorSnapshot], None]) -> DirectionMotor:
    model = DirectionMotor(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    save(model, MotorSnapshot(seed=seed, updates=0))
    for update in range(1, config.steps + 1):
        directions, move, turn = direction_batch(rng, config.batch_size)
        output = model(directions)
        move_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:, 0], move)
        turn_loss = (torch.tanh(output[:, 1]) - turn).square().mean()
        loss = move_loss + 4. * turn_loss
        if not torch.isfinite(loss):
            raise ValueError("训练损失非有限，已停止")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % config.checkpoint_every == 0 or update == config.steps:
            save(model, MotorSnapshot(seed=seed, updates=update,
                                      move_loss=float(move_loss.detach()),
                                      turn_loss=float(turn_loss.detach())))
    return model.freeze()


def run_directions(model: DirectionMotor, *, heading: float,
                   commands: list[float], segment_steps: int = 48,
                   settle_steps: int = 32) -> MotorTrial:
    if not commands or not 0 <= settle_steps < segment_steps:
        raise ValueError("方向列表不能为空，稳定观察窗口必须有效")
    env = DirectionEnvironment(heading=heading, desired=commands[0])
    initial_error = math.degrees(wrap_angle(commands[0] - heading))
    frames: list[MotorFrame] = []
    errors: list[float] = []
    progress: list[float] = []
    for command in commands:
        env.desired = command
        for local_tick in range(segment_steps):
            action = model.decide(env.observation())
            traveled = env.step(action.move, action.turn)
            error = abs(math.degrees(wrap_angle(env.desired - env.heading)))
            frames.append(MotorFrame(tick=len(frames), desired=command, heading=env.heading,
                                     x=env.x, y=env.y, move=action.move, turn=action.turn,
                                     error_degrees=error, forward_progress=traveled))
            if local_tick >= settle_steps:
                errors.append(error)
                progress.append(traveled / STEP_DISTANCE)
    maximum, fraction = max(errors), float(np.mean(progress))
    return MotorTrial(initial_error_degrees=initial_error, command_changes=len(commands) - 1,
                      steady_mean_error_degrees=float(np.mean(errors)),
                      steady_max_error_degrees=maximum,
                      steady_progress_fraction=fraction,
                      passed=maximum <= 3. and fraction >= .95, frames=frames)


def evaluate_motor(model: DirectionMotor, seed: int = 8001) -> MotorEvaluation:
    rng = np.random.default_rng(seed)
    trials = [run_directions(model, heading=0., commands=[math.radians(float(angle))])
              for angle in range(-180, 180, 5)]
    for _ in range(8):
        trials.append(run_directions(model, heading=float(rng.uniform(-math.pi, math.pi)),
                                      commands=rng.uniform(-math.pi, math.pi, 8).tolist()))
    return MotorEvaluation(trials=trials, passed=all(trial.passed for trial in trials),
                           max_steady_error_degrees=max(t.steady_max_error_degrees for t in trials),
                           min_steady_progress_fraction=min(t.steady_progress_fraction for t in trials))
