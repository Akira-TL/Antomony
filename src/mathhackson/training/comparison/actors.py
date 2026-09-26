"""神经对照共用动作与奖励训练规则，各自持有全部学习状态。"""
from __future__ import annotations

from collections import deque
import copy

import numpy as np
import torch

from mathhackson.training.direction.policy import DirectionAction, DirectionMotor
from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.reward import DIRECTIONS, actor_critic_losses, direction_distribution, discounted_returns

Policy = FeedforwardPolicy | ForagingPolicy | MemoryPolicy


def assert_reserved(model: Policy) -> None:
    if isinstance(model, (FeedforwardPolicy, MemoryPolicy)):
        model.assert_reserved()
    elif any(bool(p.any()) or p.grad is not None for p in model.reserved_parameters()):
        raise AssertionError("基础课程改变了预留接收器")


class NeuralForager:
    def __init__(self, model: Policy, motor: DirectionMotor, seed: int, *, training: bool = False) -> None:
        self.model = copy.deepcopy(model)
        self.model.set_phase("reward" if training else "frozen")
        self.motor = copy.deepcopy(motor).freeze()
        self.random = torch.Generator().manual_seed(seed)
        self.teacher_random = np.random.default_rng(seed)
        self.optimizer = (torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=.0003, weight_decay=.0001) if training else None)
        self.history: deque[torch.Tensor] = deque(maxlen=16)
        self.log_probabilities: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.updates = 0

    def restart(self, seed: int) -> None:
        if self.rewards or self.values:
            raise ValueError("上一回合奖励尚未处理")
        self.history.clear()
        self.random.manual_seed(seed)

    def act(self, observation: LocalObservation, *, sampled: bool) -> DirectionAction:
        training = self.optimizer is not None
        if training and not sampled:
            raise ValueError("当前策略梯度训练必须采样行动")
        with torch.set_grad_enabled(training):
            direction, hidden, value = self.model(torch.from_numpy(observation.vector()), tuple(self.history))
            if isinstance(self.model, (ForagingPolicy, MemoryPolicy)):
                self.history.append(hidden)
            if sampled:
                distribution = direction_distribution(direction)
                index = torch.multinomial(distribution.probs.detach(), 1, generator=self.random).squeeze(0)
                chosen = DIRECTIONS[index]
                if training:
                    self.log_probabilities.append(distribution.log_prob(index))
                    self.values.append(value)
            else:
                chosen = direction.detach()
        return self.motor.decide(chosen.numpy())

    def feedback(self, reward: float, next_observation: LocalObservation, *, terminal: bool) -> None:
        if self.optimizer is None:
            return
        self.rewards.append(reward)
        if len(self.rewards) < 32 and not terminal:
            return
        with torch.no_grad():
            bootstrap = 0. if terminal else float(self.model(torch.from_numpy(next_observation.vector()), tuple(self.history))[2])
        returns = discounted_returns(self.rewards, bootstrap, .995)
        actor, critic = actor_critic_losses(torch.stack(self.log_probabilities), torch.stack(self.values), returns)
        inputs, target = signal_batch(self.teacher_random, 64, budgets=True)
        anchor = 1. - (self.model(inputs)[0] * target).sum(-1).mean()
        loss = actor + .5 * critic + .25 * anchor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.history = deque((h.detach() for h in self.history), maxlen=16)
        self.log_probabilities.clear()
        self.values.clear()
        self.rewards.clear()
        assert_reserved(self.model)
