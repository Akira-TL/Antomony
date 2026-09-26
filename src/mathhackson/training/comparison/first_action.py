"""首次受伤前只替换一次方向；不产生在线候选或训练标签。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.revival import RevivingColony
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution
from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import AntFrame, ContinuousPlan, Frame, make_actors
from .feedback import learning_feedback
from .online_actor import UpdateRecord
from .return_reward_audit import frames


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_root: Path
    manifest: Path
    manifest_sha256: str
    protocol_sha256: str
    seeds: tuple[int, ...] = (19701, 19702)
    points_per_seed: int = 4
    branch_steps: int = 32


class Outcome(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    direction: int
    steps: int
    failures: int
    deaths: int
    exhaustion: int
    injury: float
    deliveries: int
    feedback: float
    colony_failures: int
    colony_deliveries: int


class Point(BaseModel):
    seed: int
    tick_before: int
    individual: int
    probabilities: list[float]
    initial_failures: list[int]
    initial_deaths: list[int]
    outcomes: list[Outcome]


class ParentAudit(BaseModel):
    seed: int
    steps_verified: int
    actions_verified: int
    memory_snapshots_verified: int
    points: list[tuple[int, int]]


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    elapsed_seconds: float = 0.
    plan: Plan
    source_files_verified: int
    parents: list[ParentAudit] = []


def inference_actors(plan: ContinuousPlan, seed: int) -> list[EvaluationActor]:
    return [EvaluationActor(actor.agent) for actor in make_actors(plan, seed, 'always')]


def step(env: RevivingColony, actors: list[EvaluationActor], override: tuple[int, int] | None = None,
         released: list[int] | None = None) -> Frame:
    revived = env.release_waiting() if released is None else released
    active = [not ant.exhausted for ant in env.ants]
    observations = [env.observation(i) for i in range(len(actors))]
    source = env.source_position.tolist()
    enabled = env.disturbance.active_at(env.steps)
    actions = [actor.act(observation) if alive else DirectionAction(False, 0., 0.)
               for actor, observation, alive in zip(actors, observations, active, strict=True)]
    if override is not None:
        focal, index = override
        if not 0 <= focal < len(actors) or not 0 <= index < 16 or not active[focal]:
            raise ValueError('首步方向干预无效')
        # 正常推理已推进记忆和随机流，只替换交给物理环境的动作。
        actions[focal] = actors[focal].agent.motor.decide(DIRECTIONS[index].numpy())
    injury_before = env.injuries.copy()
    events = env.step(actions)
    injury_delta = env.injuries - injury_before
    return Frame(tick=env.steps, source_position=source, source_active=enabled, food_stock=env.stock,
        ants=[AntFrame(observation=observations[i].vector().tolist(), active=active[i], move=actions[i].move,
            turn=actions[i].turn, position=ant.position.tolist(), heading=ant.heading, carrying=ant.carrying,
            exploration_left=ant.exploration_left, reserve_left=ant.reserve_left, picked_up=events[i].picked_up,
            delivered=events[i].delivered, budget_return=events[i].budget_return, exhausted=ant.exhausted,
            killed=bool(env.killed[i]), injury=float(env.injuries[i]), reward=events[i].reward, writes=0,
            learning_reward=learning_feedback('survival-v1', events[i], active=active[i],
                injury_delta=float(injury_delta[i]), injury_limit=env.disturbance.injury_limit),
            injury_delta=float(injury_delta[i]), pending=bool(env.pending[i]), respawned=i in revived,
            cumulative_deaths=int(env.deaths[i]), cumulative_terminations=int(env.terminations[i]),
            revivals=int(env.revivals[i])) for i, ant in enumerate(env.ants)])


def rollout(source: RevivingColony, parents: list[EvaluationActor], focal: int, direction: int,
            horizon: int, destination: Path) -> Outcome:
    env = copy.deepcopy(source)
    actors = [EvaluationActor(actor.agent) for actor in parents]
    initial_weights = [actor.agent.weights() for actor in actors]
    initial_deliveries = [ant.deliveries for ant in env.ants]
    initial_failures, initial_deaths = env.terminations.copy(), env.deaths.copy()
    initial_injury = env.total_injury.copy()
    feedback = 0.
    steps = 0
    with gzip.open(destination, 'xt') as stream:
        for k in range(horizon):
            if env.done:
                break
            frame = step(env, actors, (focal, direction) if k == 0 else None)
            feedback += frame.ants[focal].learning_reward
            stream.write(frame.model_dump_json() + '\n')
            steps += 1
    for actor, weights in zip(actors, initial_weights, strict=True):
        np.testing.assert_array_equal(actor.agent.weights(), weights)
    failures, deaths = env.terminations - initial_failures, env.deaths - initial_deaths
    deliveries = [ant.deliveries - old for ant, old in zip(env.ants, initial_deliveries, strict=True)]
    return Outcome(direction=direction, steps=steps, failures=int(failures[focal]), deaths=int(deaths[focal]),
        exhaustion=int(failures[focal] - deaths[focal]), injury=float(env.total_injury[focal] - initial_injury[focal]),
        deliveries=deliveries[focal], feedback=feedback, colony_failures=int(failures.sum()), colony_deliveries=sum(deliveries))


def record_point(env: RevivingColony, actors: list[EvaluationActor], seed: int, focal: int,
                 horizon: int, directory: Path) -> Point:
    directory.mkdir(parents=True, exist_ok=False)
    field_before = env.signals.trails.values.copy()
    with torch.no_grad():
        direction = actors[focal].agent.predict(torch.from_numpy(env.observation(focal).vector()),
                                               tuple(actors[focal].agent.history))[0]
        probabilities = direction_distribution(direction).probs.tolist()
    before = []
    for i, actor in enumerate(actors):
        learner = actor.agent
        history = np.asarray([h.detach().numpy() for h in learner.history], dtype=np.float32).reshape(-1, 8)
        random = learner.random.get_state().numpy().copy()
        weights = learner.weights()
        before.append((weights, history, random))
        np.savez(directory / f'initial-ant-{i:02d}.npz', weights=weights, history=history, random_state=random)
    np.savez(directory / 'initial-field.npz', values=field_before)
    outcomes = [rollout(env, actors, focal, index, horizon, directory / f'direction-{index:02d}.jsonl.gz')
                for index in range(16)]
    for actor, (weights, history, random) in zip(actors, before, strict=True):
        np.testing.assert_array_equal(actor.agent.weights(), weights)
        np.testing.assert_array_equal(np.asarray([h.numpy() for h in actor.agent.history], np.float32).reshape(-1, 8), history)
        np.testing.assert_array_equal(actor.agent.random.get_state().numpy(), random)
    np.testing.assert_array_equal(env.signals.trails.values, field_before)
    point = Point(seed=seed, tick_before=env.steps, individual=focal, probabilities=probabilities,
                  initial_failures=env.terminations.tolist(), initial_deaths=env.deaths.tolist(), outcomes=outcomes)
    (directory / 'point.json').write_text(point.model_dump_json(indent=2))
    return point


def replay_parent(plan: Plan, store: TapeStore, seed: int, output: Path) -> ParentAudit:
    continuous = store.execution.plan
    path, world = store.world('moving-danger', seed, 'always')
    condition = next(condition for condition in continuous.conditions if condition.name == 'moving-danger')
    env = RevivingColony(seed, continuous.environment, condition.disturbance)
    actors = inference_actors(continuous, seed)
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    by_key = {(record.tick, record.individual): record for record in records}
    selected: set[int] = set()
    points = []
    actions = snapshots = 0
    for expected in frames(path):
        released = env.release_waiting()
        chosen = [i for i, ant in enumerate(expected.ants) if ant.injury_delta > 0. and i not in selected]
        for focal in chosen[:max(0, plan.points_per_seed - len(selected))]:
            point = record_point(env, actors, seed, focal, plan.branch_steps,
                                 output / f'seed-{seed}/tick-{env.steps:04d}-ant-{focal:02d}')
            selected.add(focal)
            points.append((point.tick_before, focal))
        actual = step(env, actors, released=released)
        if (actual.tick, actual.source_position, actual.source_active, actual.food_stock) != (
                expected.tick, expected.source_position, expected.source_active, expected.food_stock):
            raise ValueError('父世界时钟、来源或库存重建不符')
        for i, (a, b) in enumerate(zip(actual.ants, expected.ants, strict=True)):
            if a != b.model_copy(update={'writes': 0}):
                raise ValueError(f'父世界个体状态或动作不符: {seed}/{expected.tick}/{i}')
            actions += int(a.active)
            record = by_key.get((expected.tick, i))
            if record is not None:
                np.testing.assert_array_equal(actors[i].agent.weights(), record.before)
                actors[i].agent.assign_weights(np.asarray(record.after, np.float32))
                actors[i].agent.parameter.fast = np.asarray(record.after, np.float32) - actors[i].agent.parameter.stable
            if expected.tick in world.snapshots:
                with np.load(path / f'tick-{expected.tick:04d}-ant-{i:02d}.memory.npz', allow_pickle=False) as saved:
                    history = np.asarray([h.numpy() for h in actors[i].agent.history], np.float32).reshape(-1, 8)
                    np.testing.assert_array_equal(history, saved['history'])
                    np.testing.assert_array_equal(actors[i].agent.random.get_state().numpy(), saved['random_state'])
                snapshots += 1
        if len(selected) == plan.points_per_seed or env.done:
            break
    return ParentAudit(seed=seed, steps_verified=env.steps, actions_verified=actions,
                       memory_snapshots_verified=snapshots, points=points)


def run(plan: Plan, output: Path) -> Execution:
    started = time.perf_counter()
    if plan.seeds != (19701, 19702) or plan.points_per_seed != 4 or plan.branch_steps != 32:
        raise ValueError('正式诊断只能使用已批准的两个种子、四点和32步')
    if hashlib.sha256(plan.manifest.read_bytes()).hexdigest() != plan.manifest_sha256:
        raise ValueError('输入文件清单改变')
    verified = verify_manifest(plan.source_root, plan.manifest)
    store = TapeStore(plan.source_root / 'four')
    protocol = Path('.research/protocols/four-frame-development.json').read_bytes()
    if (hashlib.sha256(protocol).hexdigest() != plan.protocol_sha256
            or store.execution.protocol_sha256 != plan.protocol_sha256
            or store.execution.plan != ContinuousPlan.model_validate_json(protocol)):
        raise ValueError('四步父世界协议身份改变')
    for source in store.execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError('源模型身份改变')
    output.mkdir(parents=True, exist_ok=False)
    execution = Execution(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        started_at=datetime.now(timezone.utc).isoformat(), plan=plan, source_files_verified=verified)
    (output / 'execution.json').write_text(execution.model_dump_json(indent=2))
    torch.set_num_threads(1)
    for seed in plan.seeds:
        execution.parents.append(replay_parent(plan, store, seed, output))
        print(execution.parents[-1].model_dump_json(), flush=True)
    execution.elapsed_seconds = time.perf_counter() - started
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (output / 'execution.json').write_text(execution.model_dump_json(indent=2))
    return execution
