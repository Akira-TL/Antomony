"""单蚁循环记忆对照：基础动作与跨步状态分开训练。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

Phase = Literal["motor", "memory", "autonomous"]
HIDDEN_WIDTH = 4
INPUT_WIDTH = 16
MOTOR_CONNECTIONS = ((0, 2, 15), (1,))


@dataclass
class Action:
    move: bool
    turn: float
    move_probability: float
    action_logp: Tensor
    entropy: Tensor
    turn_loss: Tensor


class RecurrentPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        motor = np.zeros((2, INPUT_WIDTH), np.float32)
        for row, columns in enumerate(MOTOR_CONNECTIONS):
            motor[row, list(columns)] = self.rng.normal(0, .12, len(columns))
        self.motor = torch.nn.Parameter(torch.from_numpy(motor))
        self.input_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .08, (HIDDEN_WIDTH, INPUT_WIDTH)).astype(np.float32)))
        self.hidden_weights = torch.nn.Parameter(torch.from_numpy(
            self.rng.normal(0, .08, (HIDDEN_WIDTH, HIDDEN_WIDTH)).astype(np.float32)))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(HIDDEN_WIDTH))
        self.action_weights = torch.nn.Parameter(torch.zeros(2, HIDDEN_WIDTH))
        self.parameters = (self.motor, self.input_weights, self.hidden_weights,
                           self.hidden_bias, self.action_weights)
        self.phase: Phase = "motor"
        self.hidden = torch.zeros(HIDDEN_WIDTH)
        self.outer_updates = 0
        self.outer_delta = [torch.zeros_like(value) for value in self.parameters]

    def reset_state(self) -> None:
        self.hidden = torch.zeros(HIDDEN_WIDTH)

    def trainable_masks(self) -> list[Tensor]:
        motor = torch.zeros_like(self.motor)
        if self.phase == "motor":
            for row, columns in enumerate(MOTOR_CONNECTIONS):
                motor[row, list(columns)] = 1
        memory = self.phase == "memory"
        return [motor] + [torch.full_like(value, int(memory)) for value in self.parameters[1:]]

    def decide(self, observation: np.ndarray) -> Action:
        if observation.shape != (INPUT_WIDTH,) or not np.isfinite(observation).all():
            raise ValueError("观察必须是 16 个有限值")
        training = self.phase != "autonomous"
        with torch.set_grad_enabled(training):
            output = self.motor @ torch.from_numpy(observation.astype(np.float32))
            if self.phase != "motor":
                correction = torch.tanh(self.action_weights @ self.hidden)
                output = output + correction * torch.tensor([.4, .12])
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
            target = torch.tensor((4. / 3.) * float(observation[1]), dtype=output.dtype)
            return Action(move, turn, float(move_dist.probs.detach()), logp,
                          move_dist.entropy() + turn_dist.entropy(), (output[1] - target).square())

    def observe_result(self, observation: np.ndarray) -> None:
        if observation.shape != (INPUT_WIDTH,) or not np.isfinite(observation).all():
            raise ValueError("反馈必须是 16 个有限值")
        with torch.set_grad_enabled(self.phase != "autonomous"):
            values = torch.from_numpy(observation.astype(np.float32))
            self.hidden = torch.tanh(self.input_weights @ values +
                                     self.hidden_weights @ self.hidden + self.hidden_bias)

    def finish(self, actions: list[Action], rewards: list[float], baseline: float, optimize: bool) -> None:
        self.outer_delta = [torch.zeros_like(value) for value in self.parameters]
        if optimize and self.phase != "autonomous" and actions:
            returns: list[float] = []
            future = 0.
            for reward in reversed(rewards):
                future = reward + .97 * future
                returns.append(future)
            returns.reverse()
            loss = torch.zeros(())
            for action, value in zip(actions, returns, strict=True):
                loss = loss - action.action_logp * (value - baseline) - .003 * action.entropy
                if self.phase == "motor":
                    loss = loss + 8. * action.turn_loss
            (loss / len(actions)).backward()
            masks = self.trainable_masks()
            gradients = [torch.zeros_like(value) if value.grad is None else value.grad * mask
                         for value, mask in zip(self.parameters, masks, strict=True)]
            norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in gradients))
            scale = torch.clamp(norm, min=1.)
            rate = .05 if self.phase == "motor" else .015
            with torch.no_grad():
                for parameter, gradient, mask, delta in zip(self.parameters, gradients, masks,
                                                             self.outer_delta, strict=True):
                    change = -rate * gradient / scale
                    proposal = torch.clamp(parameter + change, -6., 6.)
                    updated = torch.where(mask.bool(), proposal, parameter)
                    if not torch.isfinite(updated).all():
                        raise ValueError("非有限外部更新已拒绝")
                    delta.copy_(updated - parameter)
                    parameter.copy_(updated)
            self.outer_updates += int(any(bool(torch.count_nonzero(delta)) for delta in self.outer_delta))
        for parameter in self.parameters:
            parameter.grad = None
        self.hidden = self.hidden.detach()

    def values(self) -> list[list[list[float]]]:
        return [value.detach().reshape(value.shape[0], -1).tolist() for value in self.parameters]
