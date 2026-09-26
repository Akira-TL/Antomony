"""伤害候选的离线单次干预，保留复活；不向在线模型传回未来结果。"""
from __future__ import annotations

import copy
import gzip
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.revival import RevivingColony
from .continuous import AntFrame, Frame
from .online_actor import OnlineForager, UpdateRecord


class Outcome(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    steps: int
    focal_reward: float
    focal_deliveries: int
    focal_deaths: int
    focal_exhaustion: int
    focal_injury: float
    focal_revivals: int
    colony_deliveries: int
    colony_deaths: int


class BranchPair(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    tick: int
    individual: int
    parent_choice: UpdateRecord
    accepted_weights: list[float]
    changed: bool
    skip: Outcome
    accept: Outcome


def clone_for_branch(actors: list[OnlineForager], record: UpdateRecord, *, accept: bool) -> list[EvaluationActor]:
    if not 0 <= record.individual < len(actors):
        raise ValueError('焦点编号越界')
    source = actors[record.individual].agent
    if source.weights().tolist() != record.after or source.proposal is not None or source.awaiting_feedback:
        raise ValueError('只能从当前已处理完毕的真实提案分支')
    if record.terminal and not record.continuing_after_death:
        raise ValueError('仿真终点不能形成未来分支')
    clones = [EvaluationActor(actor.agent) for actor in actors]
    focal = clones[record.individual]
    before = np.asarray(record.before, dtype=np.float32)
    focal.agent.parameter.fast = before - focal.agent.parameter.stable
    focal.agent.assign_weights(before)
    if accept:
        focal.agent.proposal = record.proposal
        focal.changed = focal.agent.resolve(record.proposal, accept=(True,))
    return clones


def rollout(source: RevivingColony, actors: list[EvaluationActor], focal: int,
            horizon: int, path: Path) -> Outcome:
    env = copy.deepcopy(source)
    deliveries = [a.deliveries for a in env.ants]
    deaths, terminations = env.deaths.copy(), env.terminations.copy()
    injury, revivals = env.total_injury.copy(), env.revivals.copy()
    initial = env.steps
    reward = 0.
    frozen = [a.agent.weights().copy() for a in actors]
    with gzip.open(path, 'xt') as stream:
        for _ in range(horizon):
            if env.done:
                break
            revived = env.release_waiting()
            active = [not a.exhausted for a in env.ants]
            observations = [env.observation(i) for i in range(len(actors))]
            source_position = env.source_position.tolist()
            source_active = env.disturbance.active_at(env.steps)
            actions = [a.act(obs) if alive else DirectionAction(False, 0., 0.)
                       for a, obs, alive in zip(actors, observations, active, strict=True)]
            events = env.step(actions)
            reward += events[focal].reward
            frame = Frame(tick=env.steps, source_position=source_position, source_active=source_active,
                food_stock=env.stock, ants=[AntFrame(observation=observations[i].vector().tolist(),
                    active=active[i], move=actions[i].move, turn=actions[i].turn,
                    position=ant.position.tolist(), heading=ant.heading, carrying=ant.carrying,
                    exploration_left=ant.exploration_left, reserve_left=ant.reserve_left,
                    picked_up=events[i].picked_up, delivered=events[i].delivered,
                    budget_return=events[i].budget_return, exhausted=ant.exhausted, killed=bool(env.killed[i]),
                    injury=float(env.injuries[i]), reward=events[i].reward, writes=0,
                    pending=bool(env.pending[i]), respawned=i in revived,
                    cumulative_deaths=int(env.deaths[i]), cumulative_terminations=int(env.terminations[i]),
                    revivals=int(env.revivals[i])) for i, ant in enumerate(env.ants)])
            stream.write(frame.model_dump_json() + '\n')
    if any(not np.array_equal(a.agent.weights(), before) for a, before in zip(actors, frozen, strict=True)):
        raise AssertionError('分支期间发生额外参数更新')
    delivered = [a.deliveries - d for a, d in zip(env.ants, deliveries, strict=True)]
    return Outcome(steps=env.steps - initial, focal_reward=reward, focal_deliveries=delivered[focal],
        focal_deaths=int(env.deaths[focal] - deaths[focal]),
        focal_exhaustion=int(env.terminations[focal] - terminations[focal] - env.deaths[focal] + deaths[focal]),
        focal_injury=float(env.total_injury[focal] - injury[focal]),
        focal_revivals=int(env.revivals[focal] - revivals[focal]),
        colony_deliveries=sum(delivered), colony_deaths=int((env.deaths - deaths).sum()))


def evaluate(source: RevivingColony, actors: list[OnlineForager], record: UpdateRecord,
             *, horizon: int, directory: Path) -> BranchPair:
    if not 1 <= horizon <= 128 or source.done or record.tick != source.steps:
        raise ValueError('需要当前世界的非末端提案，分支最多128步')
    directory.mkdir(parents=True, exist_ok=False)
    skip = clone_for_branch(actors, record, accept=False)
    accept = clone_for_branch(actors, record, accept=True)
    focal = record.individual
    accepted_weights = accept[focal].agent.weights().tolist()
    result = BranchPair(tick=record.tick, individual=focal, parent_choice=record,
        accepted_weights=accepted_weights, changed=accept[focal].changed,
        skip=rollout(source, skip, focal, horizon, directory / 'skip.jsonl.gz'),
        accept=rollout(source, accept, focal, horizon, directory / 'accept.jsonl.gz'))
    (directory / 'pair.json').write_text(result.model_dump_json(indent=2))
    return result
