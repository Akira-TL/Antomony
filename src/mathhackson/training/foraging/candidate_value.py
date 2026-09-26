"""训练期同状态分支评价；未来结果不进入在线行动或主轨迹。"""
from __future__ import annotations

from collections import deque
import copy

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.direction.policy import DirectionAction
from .adaptation import NovelSignalLearner
from .disturbance import DisturbedColony
from .environment import LocalObservation
from .reward import DIRECTIONS, direction_distribution


class BranchOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    steps: int
    focal_reward: float
    colony_reward: float
    focal_deliveries: int
    colony_deliveries: int
    focal_death: bool
    colony_deaths: int
    focal_exhausted: bool
    focal_injury: float


class CandidateValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    skip: BranchOutcome
    accept: BranchOutcome
    changed: bool
    accepted_weights: list[float]


class EvaluationActor:
    """复制推理状态，不复制仍附着计算图的训练缓冲。"""

    def __init__(self, source: NovelSignalLearner, *, accept: bool = False) -> None:
        if source.awaiting_feedback:
            raise ValueError("分支必须在实际行动反馈完成后创建")
        if accept and (source.proposal is None or source.terminal):
            raise ValueError("只能接受存活个体的当前待定提案")
        self.agent = type(source)(source.policy, source.motor, 0, source.config)
        self.agent.parameter = copy.deepcopy(source.parameter)
        self.agent.assign_weights(source.weights())
        self.agent.random.set_state(source.random.get_state().clone())
        self.agent.history = deque((h.detach().clone() for h in source.history), maxlen=16)
        self.changed = False
        if accept:
            self.agent.proposal = source.proposal
            self.changed = self.agent.resolve(source.proposal, accept=(True,) * len(source.adaptive_parameters()))

    @torch.no_grad()
    def act(self, observation: LocalObservation) -> DirectionAction:
        direction, hidden, _ = self.agent.predict(torch.from_numpy(observation.vector()), tuple(self.agent.history))
        probabilities = direction_distribution(direction).probs
        index = torch.multinomial(probabilities, 1, generator=self.agent.random).squeeze(0)
        self.agent.history.append(hidden)
        return self.agent.motor.decide(DIRECTIONS[index].numpy().copy())


def _rollout(env: DisturbedColony, actors: list[EvaluationActor], focal: int, horizon: int) -> BranchOutcome:
    initial_tick = env.steps
    initial_deliveries = [ant.deliveries for ant in env.ants]
    initial_killed = env.killed.copy()
    initial_injury = float(env.injuries[focal])
    focal_reward = colony_reward = 0.
    for _ in range(horizon):
        if env.done:
            break
        actions = [actor.act(env.observation(i)) if not env.ants[i].exhausted else DirectionAction(False, 0., 0.)
                   for i, actor in enumerate(actors)]
        events = env.step(actions)
        focal_reward += events[focal].reward
        colony_reward += sum(event.reward for event in events)
    deliveries = [ant.deliveries - before for ant, before in zip(env.ants, initial_deliveries, strict=True)]
    return BranchOutcome(steps=env.steps - initial_tick, focal_reward=focal_reward, colony_reward=colony_reward,
                         focal_deliveries=deliveries[focal], colony_deliveries=sum(deliveries),
                         focal_death=bool(env.killed[focal] and not initial_killed[focal]),
                         colony_deaths=int(np.count_nonzero(env.killed & ~initial_killed)),
                         focal_exhausted=env.ants[focal].exhausted,
                         focal_injury=float(env.injuries[focal]) - initial_injury)


def evaluate_candidate(env: DisturbedColony, agents: list[NovelSignalLearner], focal: int,
                       *, horizon: int = 64) -> CandidateValue:
    if horizon < 1 or not 0 <= focal < len(agents) or len(agents) != len(env.ants):
        raise ValueError("分支时限和个体索引必须有效")
    if env.done or env.ants[focal].exhausted or agents[focal].terminal or agents[focal].proposal is None:
        raise ValueError("这里只评价有待定更新的存活个体，不把下一代混入短期标签")
    skip = [EvaluationActor(agent) for agent in agents]
    accept = [EvaluationActor(agent, accept=i == focal) for i, agent in enumerate(agents)]
    return CandidateValue(skip=_rollout(copy.deepcopy(env), skip, focal, horizon),
                          accept=_rollout(copy.deepcopy(env), accept, focal, horizon),
                          changed=accept[focal].changed,
                          accepted_weights=accept[focal].agent.weights().tolist())
