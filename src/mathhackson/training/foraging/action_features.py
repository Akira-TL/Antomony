"""以相同当前观察预览候选动作分布，不执行环境或消费随机数。"""
from __future__ import annotations

import numpy as np
import torch

from mathhackson.training.direction.environment import MAX_TURN
from .environment import LocalObservation
from .reward import DIRECTIONS
from .trust_candidate import TrustDirectionLearner, rotation_probabilities

ACTION_FEATURE_WIDTH = 38


def action_effect_features(agent: TrustDirectionLearner, observation: LocalObservation) -> np.ndarray:
    proposal = agent.proposal
    if proposal is None or agent.awaiting_feedback:
        raise ValueError('动作预览需要已取得反馈的当前候选')
    after = agent.preview_weights(proposal, accept=(True,))
    tensor = torch.from_numpy(observation.vector())
    with torch.no_grad():
        base = agent.policy(tensor, tuple(agent.history))[0].unsqueeze(0)
        novel = tensor[:72].reshape(9, 8)[:, 3:].flatten().unsqueeze(0)
        before_p = rotation_probabilities(torch.from_numpy(agent.weights()), base, novel)[0].numpy()
        after_p = rotation_probabilities(torch.from_numpy(after), base, novel)[0].numpy()
    actions = [agent.motor.decide(direction.numpy()) for direction in DIRECTIONS]
    moves = np.asarray([a.move for a in actions], dtype=np.float64)
    turns = np.asarray([a.turn for a in actions], dtype=np.float64)
    headings = np.stack((np.cos(turns * MAX_TURN), np.sin(turns * MAX_TURN)), axis=-1)
    difference = after_p.astype(np.float64) - before_p
    displacement = (difference * moves) @ headings
    moments = np.asarray((before_p @ moves, after_p @ moves, before_p @ turns, after_p @ turns,
                          displacement[0], displacement[1]))
    values = np.concatenate((before_p, difference, moments))
    if values.shape != (ACTION_FEATURE_WIDTH,) or not np.isfinite(values).all():
        raise ValueError('候选动作特征无效')
    return (values / (1. + np.abs(values))).astype(np.float32)
