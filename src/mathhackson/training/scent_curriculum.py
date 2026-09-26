"""Supervise local scent selection without exposing a global destination at inference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .recurrent import SAFETY_PARAMETER_LIMIT
from .roundtrip_policy import RoundTripPolicy


@dataclass(frozen=True)
class ScentMetrics:
    turn_error: float
    conflict_accuracy: float
    conflict_samples: int


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
