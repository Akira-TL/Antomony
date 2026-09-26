"""Supervise local scent selection without exposing a global destination at inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .recurrent import SAFETY_PARAMETER_LIMIT
from .roundtrip_environment import RoundTripEnvironment
from .roundtrip_policy import RoundTripPolicy


@dataclass(frozen=True)
class ScentMetrics:
    turn_error: float
    conflict_accuracy: float
    conflict_samples: int


@dataclass(frozen=True)
class MemoryScore:
    turn_error: float
    direction_accuracy: float
    samples: int


@dataclass(frozen=True)
class MemoryMetrics:
    full: MemoryScore
    without_memory: MemoryScore
    short_only: MemoryScore
    long_only: MemoryScore


@dataclass(frozen=True)
class TeacherMetrics:
    turn_error: float
    return_error: float
    cue_missing_error: float
    return_steps: int


@dataclass(frozen=True)
class TeacherTrajectory:
    observations: np.ndarray
    feedback: np.ndarray
    target_turns: np.ndarray


def _examples(rng: np.random.Generator, count: int) -> tuple[Tensor, Tensor, np.ndarray]:
    observations = np.zeros((count, 17), np.float32)
    angles = rng.uniform(-math.pi, math.pi, (2, count))
    magnitudes = rng.uniform(.08, 1., (2, count))
    for channel, start in enumerate((9, 12)):
        observations[:, start] = magnitudes[channel] * np.cos(angles[channel])
        observations[:, start + 1] = magnitudes[channel] * np.sin(angles[channel])
        observations[:, start - 1] = rng.uniform(0., 1., count)
    carrying = rng.integers(0, 2, count).astype(np.float32)
    observations[:, 16] = carrying
    observations[:, 15] = 1.
    observations[:, 3] = rng.integers(0, 2, count)
    observations[:, 4] = rng.uniform(-1., 1., count)
    observations[:, 14] = rng.uniform(0., 1., count)
    active = np.where(carrying > .5, angles[0], angles[1])
    desired_turn = np.clip(active / math.radians(10.), -.95, .95).astype(np.float32)
    conflict = np.sign(np.sin(angles[0])) != np.sign(np.sin(angles[1]))
    return torch.from_numpy(observations), torch.from_numpy(desired_turn), conflict


def _turn_mean(model: RoundTripPolicy, observations: Tensor) -> Tensor:
    hidden = torch.tanh(.5 * torch.nn.functional.layer_norm(
        observations @ model.input_weights.T + model.hidden_bias, (model.hidden.numel(),)))
    output = observations @ model.motor.T + torch.tanh(hidden @ model.action_weights.T) * \
        observations.new_tensor(model.correction_scale)
    return 1.5 * torch.tanh(output[:, 1] / 1.5)


def evaluate_scent_reader(model: RoundTripPolicy, *, seed: int = 20260927,
                          samples: int = 2048) -> ScentMetrics:
    observations, desired, conflict = _examples(np.random.default_rng(seed), samples)
    with torch.no_grad():
        turns = torch.tanh(3. * _turn_mean(model, observations)).numpy()
    goals = desired.numpy()
    selected = conflict & (np.abs(goals) > .5)
    return ScentMetrics(
        turn_error=float(np.mean(np.abs(turns - goals))),
        conflict_accuracy=float(np.mean(np.sign(turns[selected]) == np.sign(goals[selected]))),
        conflict_samples=int(selected.sum()),
    )


def pretrain_scent_reader(model: RoundTripPolicy, *, seed: int = 20260926,
                          steps: int = 300, batch_size: int = 256,
                          checkpoint_every: int = 50,
                          checkpoint: Callable[[int, ScentMetrics], None] | None = None
                          ) -> ScentMetrics:
    if steps < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("training steps, batch size, and checkpoint interval must be positive")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam((model.input_weights, model.action_weights), lr=.02)
    mask = torch.zeros_like(model.input_weights)
    mask[:, 8:14] = 1.
    mask[:, 16] = 1.
    for step in range(1, steps + 1):
        observations, desired, _ = _examples(rng, batch_size)
        target_mean = torch.atanh(desired) / 3.
        loss = (_turn_mean(model, observations) - target_mean).square().mean()
        optimizer.zero_grad()
        loss.backward()
        assert model.input_weights.grad is not None
        model.input_weights.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_((model.input_weights, model.action_weights), 1.)
        optimizer.step()
        with torch.no_grad():
            model.input_weights.clamp_(-SAFETY_PARAMETER_LIMIT, SAFETY_PARAMETER_LIMIT)
            model.action_weights.clamp_(-SAFETY_PARAMETER_LIMIT, SAFETY_PARAMETER_LIMIT)
        if checkpoint is not None and (step % checkpoint_every == 0 or step == steps):
            checkpoint(step, evaluate_scent_reader(model))
    return evaluate_scent_reader(model)


def _memory_examples(rng: np.random.Generator, count: int, length: int
                     ) -> tuple[Tensor, Tensor, Tensor]:
    if count < 2 or count % 2 or length < 17:
        raise ValueError("记忆样本数须为正偶数，序列至少 17 步")
    pairs = count // 2
    observations = np.zeros((length, count, 17), np.float32)
    targets = np.zeros((length, count), np.float32)
    queries = np.zeros((length, count), np.bool_)
    queries[[4, 8, 12, 16]] = True
    bearings = rng.uniform(.9, 2.4, pairs)
    target_heading = np.concatenate((bearings, -bearings))
    carrying = np.tile(rng.integers(0, 2, pairs).astype(np.float32), 2)
    rotations = np.tile(rng.uniform(-.3, .3, (length, pairs)), (1, 2))
    heading = np.zeros(count)
    rows = np.arange(count)
    active_channel = np.where(carrying > .5, 0, 1)
    other_channel = 1 - active_channel
    for tick in range(length):
        if tick:
            heading += rotations[tick - 1] * math.radians(10.)
        relative = np.arctan2(np.sin(target_heading - heading),
                              np.cos(target_heading - heading))
        targets[tick] = np.clip(relative / math.radians(10.), -.95, .95)
        current = observations[tick]
        current[:, 3] = float(tick > 0)
        current[:, 4] = rotations[tick - 1] if tick else 0.
        current[:, 14] = tick / length
        current[:, 15] = 1.
        current[:, 16] = carrying
        distractor_angle = np.tile(rng.uniform(-math.pi, math.pi, pairs), 2)
        distractor_strength = np.tile(rng.uniform(.08, 1., pairs), 2)
        current[rows, 9 + other_channel * 3] = distractor_strength * np.cos(distractor_angle)
        current[rows, 10 + other_channel * 3] = distractor_strength * np.sin(distractor_angle)
        current[rows, 8 + other_channel * 3] = distractor_strength
        if tick == 0:
            strength = np.tile(rng.uniform(.08, 1., pairs), 2)
            current[rows, 9 + active_channel * 3] = strength * np.cos(relative)
            current[rows, 10 + active_channel * 3] = strength * np.sin(relative)
            current[rows, 8 + active_channel * 3] = strength
    return torch.from_numpy(observations), torch.from_numpy(targets), torch.from_numpy(queries)


def _memory_turns(model: RoundTripPolicy, observations: Tensor,
                  active_taps: tuple[bool, ...] | None = None,
                  feedback: Tensor | None = None) -> Tensor:
    if active_taps is None:
        active_taps = (True,) * len(model.memory_lags)
    if len(active_taps) != len(model.memory_lags):
        raise ValueError("记忆位置掩码长度不匹配")
    if feedback is not None and feedback.shape != observations.shape:
        raise ValueError("行动与反馈序列形状不匹配")
    batch = observations.shape[1]
    history: list[Tensor] = []
    hidden = observations.new_zeros((batch, model.hidden.numel()))
    turns = []
    for tick, values in enumerate(observations):
        output = values @ model.motor.T + torch.tanh(hidden @ model.action_weights.T) * \
            values.new_tensor(model.correction_scale)
        turns.append(1.5 * torch.tanh(output[:, 1] / 1.5))
        taps = torch.cat(tuple(history[-lag] if enabled and len(history) >= lag else
                               values.new_zeros((batch, model.hidden.numel()))
                               for lag, enabled in zip(model.memory_lags, active_taps, strict=True)), dim=1)
        after_action = feedback[tick] if feedback is not None else values
        hidden = torch.tanh(.5 * torch.nn.functional.layer_norm(
            after_action @ model.input_weights.T + taps @ model.hidden_weights.T + model.hidden_bias,
            (model.hidden.numel(),)))
        history.append(hidden)
        history = history[-max(model.memory_lags):]
    return torch.stack(turns)


def evaluate_scent_memory(model: RoundTripPolicy, *, seed: int = 20260928,
                          samples: int = 256, length: int = 20) -> MemoryMetrics:
    observations, desired, queries = _memory_examples(np.random.default_rng(seed), samples, length)
    target = desired[queries].numpy()

    def score(taps: tuple[bool, ...]) -> MemoryScore:
        with torch.no_grad():
            turns = torch.tanh(3. * _memory_turns(model, observations, taps))[queries].numpy()
        return MemoryScore(float(np.mean(np.abs(turns - target))),
                           float(np.mean(np.sign(turns) == np.sign(target))), len(target))

    return MemoryMetrics(
        full=score((True,) * 8),
        without_memory=score((False,) * 8),
        short_only=score((True,) * 4 + (False,) * 4),
        long_only=score((False,) * 4 + (True,) * 4),
    )


def pretrain_scent_memory(model: RoundTripPolicy, *, seed: int = 20260929,
                          steps: int = 200, batch_size: int = 64, length: int = 20,
                          checkpoint_every: int = 50,
                          checkpoint: Callable[[int, MemoryMetrics], None] | None = None
                          ) -> MemoryMetrics:
    if min(steps, checkpoint_every) < 1 or batch_size < 2 or batch_size % 2 or length < 17:
        raise ValueError("训练次数须为正，样本数须为正偶数，序列至少 17 步")
    if model.phase != "memory":
        raise ValueError("间断气味预训练只允许在记忆阶段执行")
    rng = np.random.default_rng(seed)
    parameters = (model.input_weights, model.hidden_weights, model.hidden_bias,
                  model.action_weights)
    masks = model.trainable_masks()[1:5]
    optimizer = torch.optim.Adam(parameters, lr=.01)
    for step in range(1, steps + 1):
        observations, desired, queries = _memory_examples(rng, batch_size, length)
        target_mean = torch.atanh(desired) / 3.
        error = (_memory_turns(model, observations) - target_mean).square()
        loss = error[queries].mean() + .2 * error[1].mean()
        optimizer.zero_grad()
        loss.backward()
        for parameter, mask in zip(parameters, masks, strict=True):
            assert parameter.grad is not None
            parameter.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters:
                parameter.clamp_(-SAFETY_PARAMETER_LIMIT, SAFETY_PARAMETER_LIMIT)
        if checkpoint is not None and (step % checkpoint_every == 0 or step == steps):
            checkpoint(step, evaluate_scent_memory(model))
    optimizer.zero_grad(set_to_none=True)
    return evaluate_scent_memory(model)


def _teacher_trajectory(seed: int) -> TeacherTrajectory:
    environment = RoundTripEnvironment(seed)
    observations = []
    feedback = []
    turns = []
    while environment.delivered < 1 and not environment.done:
        observation = environment.observation()
        target = environment.home if environment.carrying else environment.food
        offset = target - environment.position
        angle = math.atan2(float(offset[1]), float(offset[0])) - environment.heading
        relative = math.atan2(math.sin(angle), math.cos(angle))
        turn = float(np.clip(relative / math.radians(environment.max_turn), -.95, .95))
        carrying = environment.carrying
        environment.step(True, turn, not carrying, carrying)
        observations.append(observation)
        feedback.append(environment.observation())
        turns.append(turn)
    if environment.delivered != 1:
        raise RuntimeError(f"教师轨迹未完成首次交付：{seed}")
    return TeacherTrajectory(np.asarray(observations, np.float32),
                             np.asarray(feedback, np.float32), np.asarray(turns, np.float32))


def _teacher_batch(trajectories: list[TeacherTrajectory]
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    length = max(len(item.target_turns) for item in trajectories)
    count = len(trajectories)
    observations = np.zeros((length, count, 17), np.float32)
    feedback = np.zeros_like(observations)
    targets = np.zeros((length, count), np.float32)
    valid = np.zeros((length, count), np.bool_)
    for column, item in enumerate(trajectories):
        size = len(item.target_turns)
        observations[:size, column] = item.observations
        feedback[:size, column] = item.feedback
        targets[:size, column] = item.target_turns
        valid[:size, column] = True
    return (torch.from_numpy(observations), torch.from_numpy(feedback),
            torch.from_numpy(targets), torch.from_numpy(valid))


def evaluate_teacher_scent(model: RoundTripPolicy, *, first_seed: int = 2000,
                           seeds: int = 16) -> TeacherMetrics:
    if seeds < 1:
        raise ValueError("验收轨迹数须为正")
    trajectories = [_teacher_trajectory(seed) for seed in range(first_seed, first_seed + seeds)]
    observations, feedback, targets, valid = _teacher_batch(trajectories)
    with torch.no_grad():
        turns = torch.tanh(3. * _memory_turns(model, observations, feedback=feedback))
        error = (turns - targets).abs()
        returning = valid & (observations[:, :, 16] > .5)
        missing = returning & (observations[:, :, 8] < .005)
        return TeacherMetrics(float(error[valid].mean()), float(error[returning].mean()),
                              float(error[missing].mean()), int(returning.sum()))


def pretrain_teacher_scent(model: RoundTripPolicy, *, steps: int = 100,
                           train_seed: int = 1000, train_seeds: int = 64,
                           batch_size: int = 8, checkpoint_every: int = 25,
                           checkpoint: Callable[[int, TeacherMetrics], None] | None = None
                           ) -> TeacherMetrics:
    if min(steps, train_seeds, batch_size, checkpoint_every) < 1:
        raise ValueError("训练次数、轨迹数、批量和快照间隔须为正")
    if model.phase != "memory":
        raise ValueError("真实轨迹预训练只允许在记忆阶段执行")
    trajectories = [_teacher_trajectory(seed) for seed in range(train_seed, train_seed + train_seeds)]
    rng = np.random.default_rng(20260930)
    parameters = (model.input_weights, model.hidden_weights, model.hidden_bias,
                  model.action_weights)
    masks = model.trainable_masks()[1:5]
    optimizer = torch.optim.Adam(parameters, lr=.003)
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(trajectories), batch_size)
        observations, feedback, desired, valid = _teacher_batch([trajectories[index] for index in indices])
        target_mean = torch.atanh(desired) / 3.
        error = (_memory_turns(model, observations, feedback=feedback) - target_mean).square()
        weights = valid * torch.where(observations[:, :, 16] > .5, 3., 1.)
        loss = (error * weights).sum() / weights.sum()
        optimizer.zero_grad()
        loss.backward()
        for parameter, mask in zip(parameters, masks, strict=True):
            assert parameter.grad is not None
            parameter.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters:
                parameter.clamp_(-SAFETY_PARAMETER_LIMIT, SAFETY_PARAMETER_LIMIT)
        if checkpoint is not None and (step % checkpoint_every == 0 or step == steps):
            checkpoint(step, evaluate_teacher_scent(model))
    optimizer.zero_grad(set_to_none=True)
    return evaluate_teacher_scent(model)
