"""单蚁循环记忆与受限的反馈后动作参数写入。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Phase = Literal["motor", "memory", "adaptive", "autonomous"]
WriteMode = Literal["off", "learned", "always"]
HIDDEN_WIDTH = 8
MEMORY_LAGS = (1, 8, 12, 16)
MODEL_VERSION = "sparse-memory-v2"
INPUT_WIDTH = 16
MOTOR_CONNECTIONS = ((0, 2, 15), (1,))
FAST_LIMITS = torch.tensor([.3, .15, .12])
FAST_STEPS = torch.tensor([.025, .012, .012])
SAFETY_PARAMETER_LIMIT = 10.


@dataclass
class Action:
    move: bool
    turn: float
    move_probability: float
    action_logp: Tensor
    entropy: Tensor
    move_loss: Tensor
    turn_loss: Tensor
    release_home: bool = False
    release_food: bool = False
    release_home_probability: float = 0.
    release_food_probability: float = 0.


@dataclass
class Write:
    requested: bool
    wrote: bool
    probability: float
    logp: Tensor


class RecurrentPolicy:
    correction_scale = (.4, .12)

    def __init__(self, seed: int, input_width: int = INPUT_WIDTH) -> None:
        self.rng = np.random.default_rng(seed)
        self.input_width = input_width
        motor = np.zeros((2, input_width), np.float32)
        for row, columns in enumerate(MOTOR_CONNECTIONS):
            motor[row, list(columns)] = self.rng.normal(0, .12, len(columns))
        self.motor = torch.nn.Parameter(torch.from_numpy(motor))
        self.input_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .08, (HIDDEN_WIDTH, input_width)).astype(np.float32)))
        self.hidden_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .08, (HIDDEN_WIDTH, HIDDEN_WIDTH * len(MEMORY_LAGS))).astype(np.float32)))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(HIDDEN_WIDTH))
        self.action_weights = torch.nn.Parameter(torch.zeros(2, HIDDEN_WIDTH))
        self.gate_weights = torch.nn.Parameter(torch.zeros(HIDDEN_WIDTH + 1))
        with torch.no_grad():
            self.gate_weights[-1] = -2.
        self.write_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .05, (3, HIDDEN_WIDTH + 1)).astype(np.float32)))
        self.parameters = (self.motor, self.input_weights, self.hidden_weights,
                           self.hidden_bias, self.action_weights, self.gate_weights,
                           self.write_weights)
        self.phase: Phase = "motor"
        self.write_mode: WriteMode = "off"
        self.hidden = torch.zeros(HIDDEN_WIDTH)
        self.hidden_history: list[Tensor] = []
        self.fast = torch.zeros(3)
        self.fast_delta = torch.zeros(3)
        self.self_updates = 0
        self.outer_updates = 0
        self.outer_delta = [torch.zeros_like(value) for value in self.parameters]

    def reset_state(self) -> None:
        self.hidden = torch.zeros(HIDDEN_WIDTH)
        self.hidden_history = []
        self.fast = torch.zeros(3)
        self.fast_delta = torch.zeros(3)

    def trainable_masks(self) -> list[Tensor]:
        motor = torch.zeros_like(self.motor)
        if self.phase == "motor":
            for row, columns in enumerate(MOTOR_CONNECTIONS):
                motor[row, list(columns)] = 1
        memory = self.phase in {"memory", "adaptive"}
        writing = self.phase == "adaptive"
        return [motor] + [torch.full_like(value, int(memory)) for value in self.parameters[1:5]] + [
            torch.full_like(value, int(writing)) for value in self.parameters[5:]]

    def decide(self, observation: np.ndarray) -> Action:
        if observation.shape != (self.input_width,) or not np.isfinite(observation).all():
            raise ValueError(f"观察必须是 {self.input_width} 个有限值")
        training = self.phase != "autonomous"
        with torch.set_grad_enabled(training):
            output = self.motor @ torch.from_numpy(observation.astype(np.float32))
            if self.phase != "motor":
                correction = torch.tanh(self.action_weights @ self.hidden)
                output = output + correction * torch.tensor(self.correction_scale)
                if self.write_mode != "off":
                    output = output + torch.stack((self.fast[0],
                                                   self.fast[1] * float(observation[1]) + self.fast[2]))
            output = torch.stack((3. * torch.tanh(output[0] / 3.),
                                  1.5 * torch.tanh(output[1] / 1.5)))
            move_dist = torch.distributions.Bernoulli(logits=output[0])
            turn_dist = torch.distributions.Normal(output[1], .35)
            if training:
                move = bool(self.rng.random() < float(move_dist.probs.detach()))
                raw_turn = float(self.rng.normal(float(output[1].detach()), .35))
            else:
                move = bool(move_dist.probs >= .5)
                raw_turn = float(output[1])
            turn = float(np.tanh(3. * raw_turn))
            logp = move_dist.log_prob(torch.tensor(float(move))) + turn_dist.log_prob(torch.tensor(raw_turn))
            move_target = torch.tensor(float(observation[0] > .25), dtype=output.dtype)
            move_loss = torch.nn.functional.binary_cross_entropy_with_logits(output[0], move_target)
            target = torch.tensor((4. / 3.) * float(observation[1]), dtype=output.dtype)
            return Action(move, turn, float(move_dist.probs.detach()), logp,
                          move_dist.entropy() + turn_dist.entropy(), move_loss,
                          (output[1] - target).square())

    def observe_result(self, observation: np.ndarray, *, terminal: bool = False) -> Write:
        if observation.shape != (self.input_width,) or not np.isfinite(observation).all():
            raise ValueError(f"反馈必须是 {self.input_width} 个有限值")
        with torch.set_grad_enabled(self.phase != "autonomous"):
            values = torch.from_numpy(observation.astype(np.float32))
            taps = torch.cat(tuple(self.hidden_history[-lag] if len(self.hidden_history) >= lag
                                   else torch.zeros_like(self.hidden) for lag in MEMORY_LAGS))
            preactivation = self.input_weights @ values + self.hidden_weights @ taps + self.hidden_bias
            self.hidden = torch.tanh(.5 * torch.nn.functional.layer_norm(preactivation, (HIDDEN_WIDTH,)))
            self.hidden_history.append(self.hidden)
            self.hidden_history = self.hidden_history[-max(MEMORY_LAGS):]
            self.fast_delta = torch.zeros_like(self.fast)
            if terminal or self.write_mode == "off" or self.phase in {"motor", "memory"}:
                return Write(False, False, 0., torch.zeros(()))
            features = torch.cat((self.hidden, torch.ones(1)))
            gate = torch.distributions.Bernoulli(logits=self.gate_weights @ features)
            probability = float(gate.probs.detach())
            requested = (True if self.write_mode == "always" else
                         bool(self.rng.random() < probability) if self.phase == "adaptive" else
                         probability >= .5)
            gate_logp = gate.log_prob(torch.tensor(float(requested))) if self.write_mode == "learned" else torch.zeros(())
            wrote = False
            if requested:
                proposal = torch.clamp(self.fast + torch.tanh(self.write_weights @ features) * FAST_STEPS,
                                       -FAST_LIMITS, FAST_LIMITS)
                self.fast_delta = (proposal - self.fast).detach()
                wrote = bool(torch.count_nonzero(self.fast_delta))
                self.fast = proposal
                self.self_updates += int(wrote)
            return Write(requested, wrote, probability, gate_logp)

    def finish(self, actions: list[Action], writes: list[Write], rewards: list[float],
               baseline: float, optimize: bool) -> None:
        self.outer_delta = [torch.zeros_like(value) for value in self.parameters]
        if optimize and self.phase != "autonomous" and actions:
            returns: list[float] = []
            future = 0.
            for reward in reversed(rewards):
                future = reward + .97 * future
                returns.append(future)
            returns.reverse()
            loss = torch.zeros(())
            for index, (action, write, value) in enumerate(zip(actions, writes, returns, strict=True)):
                loss = loss - action.action_logp * (value - baseline) - .003 * action.entropy
                if self.phase == "motor":
                    loss = loss + 4. * action.move_loss + 8. * action.turn_loss
                if self.phase == "adaptive" and self.write_mode == "learned" and index + 1 < len(returns):
                    future = returns[index + 1] - .02 * int(write.requested)
                    loss = loss - write.logp * (future - baseline)
            masks = self.trainable_masks()
            loss = loss / len(actions) + .001 * sum(
                (parameter * mask).square().sum() / mask.sum().clamp(min=1.)
                for parameter, mask in zip(self.parameters, masks, strict=True))
            loss.backward()
            gradients = [torch.zeros_like(value) if value.grad is None else value.grad * mask
                         for value, mask in zip(self.parameters, masks, strict=True)]
            norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in gradients))
            scale = torch.clamp(norm, min=1.)
            rate = .05 if self.phase == "motor" else .015
            with torch.no_grad():
                for parameter, gradient, mask, delta in zip(
                        self.parameters, gradients, masks, self.outer_delta, strict=True):
                    change = -rate * gradient / scale
                    proposal = torch.clamp(parameter + change,
                                           -SAFETY_PARAMETER_LIMIT, SAFETY_PARAMETER_LIMIT)
                    updated = torch.where(mask.bool(), proposal, parameter)
                    if not torch.isfinite(updated).all():
                        raise ValueError("非有限外部更新已拒绝")
                    delta.copy_(updated - parameter)
                    parameter.copy_(updated)
            self.outer_updates += int(any(bool(torch.count_nonzero(delta)) for delta in self.outer_delta))
        for parameter in self.parameters:
            parameter.grad = None
        self.hidden = self.hidden.detach()
        self.fast = self.fast.detach()

    def values(self) -> list[list[list[float]]]:
        return [value.detach().reshape(value.shape[0], -1).tolist() for value in self.parameters]
