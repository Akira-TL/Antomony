"""基础奖励课程的方向采样与有限步演员-评论家损失。"""
from __future__ import annotations

import math

import torch
from torch.distributions import Categorical

ANGLES = torch.arange(16, dtype=torch.float32) * (2. * math.pi / 16.)
DIRECTIONS = torch.stack((torch.cos(ANGLES), torch.sin(ANGLES)), dim=-1)


def direction_distribution(direction: torch.Tensor) -> Categorical:
    scores = 8. * (direction @ DIRECTIONS.T)
    probabilities = .8 * torch.softmax(scores, dim=-1) + .2 / 16.
    return Categorical(probs=probabilities)


def discounted_returns(rewards: list[float], bootstrap: float, gamma: float = .995) -> torch.Tensor:
    if not rewards or not 0. <= gamma <= 1. or not all(math.isfinite(x) for x in (*rewards, bootstrap)):
        raise ValueError("奖励、折扣或后续价值无效")
    result = []
    value = bootstrap
    for reward in reversed(rewards):
        value = reward + gamma * value
        result.append(value)
    return torch.tensor(result[::-1], dtype=torch.float32)


def actor_critic_losses(log_probabilities: torch.Tensor, values: torch.Tensor,
                        returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = returns - values.detach()
    actor = -(log_probabilities * advantage).mean()
    critic = torch.nn.functional.smooth_l1_loss(values, returns)
    return actor, critic
