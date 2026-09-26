"""重建已登记主轨迹，按过去伤害选点；只在隔离副本执行未来分支。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.revival import RevivingColony
from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import Frame, make_actors, save_actors
from .injury_branch import evaluate
from .online_actor import OnlineForager, UpdateRecord


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_root: Path
    manifest: Path
    seeds: tuple[int, ...] = (19301, 19302)
    maximum_parent_steps: int = Field(default=1536, ge=1)
    pairs_per_world: int = Field(default=16, ge=1, le=32)
    branch_steps: int = Field(default=128, ge=1, le=128)


class ProbeWorld(BaseModel):
    seed: int
    parent_steps: int
    points: list[tuple[int, int]]


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    plan: Plan
    verified_source_files: int
    worlds: list[ProbeWorld] = []


def probe_world(plan: Plan, store: TapeStore, seed: int, output: Path) -> ProbeWorld:
    continuous = store.execution.plan
    source, _ = store.world('moving-danger', seed, 'learned')
    condition = next(c for c in continuous.conditions if c.name == 'moving-danger')
    env = RevivingColony(seed, continuous.environment, condition.disturbance)
    actors = make_actors(continuous, seed, 'learned')
    if not all(isinstance(a, OnlineForager) for a in actors):
        raise TypeError('诊断要求在线个体')
    records = [UpdateRecord.model_validate_json(line) for line in (source / 'updates.jsonl').read_text().splitlines()]
    by_key = {(r.tick, r.individual): r for r in records}
    selected: set[int] = set()
    points: list[tuple[int, int]] = []
    damage = np.zeros(continuous.environment.ants)
    with gzip.open(source / 'trajectory.jsonl.gz', 'rt') as trace:
        for line in trace:
            expected = Frame.model_validate_json(line)
            revived = env.release_waiting()
            for i in revived:
                actors[i].revive()
            active = [not a.exhausted for a in env.ants]
            observations = [env.observation(i) for i in range(len(actors))]
            assert expected.source_position == env.source_position.tolist()
            assert expected.source_active == condition.disturbance.active_at(env.steps)
            actions = []
            for i, actor in enumerate(actors):
                np.testing.assert_array_equal(observations[i].vector(), expected.ants[i].observation)
                action = actor.act(observations[i]) if active[i] else DirectionAction(False, 0., 0.)
                assert action.move == expected.ants[i].move and action.turn == expected.ants[i].turn
                actions.append(action)
            injury_before = env.total_injury.copy()
            events = env.step(actions)
            damage += env.total_injury - injury_before
            pending: list[UpdateRecord] = []
            for i, actor in enumerate(actors):
                recorded = expected.ants[i]
                ant, event = env.ants[i], events[i]
                assert ant.position.tolist() == recorded.position and ant.heading == recorded.heading
                assert ant.carrying == recorded.carrying and ant.exhausted == recorded.exhausted
                assert event.reward == recorded.reward
                assert (event.picked_up, event.delivered, event.budget_return) == (recorded.picked_up, recorded.delivered, recorded.budget_return)
                assert (int(env.deaths[i]), int(env.terminations[i]), int(env.revivals[i])) == (
                    recorded.cumulative_deaths, recorded.cumulative_terminations, recorded.revivals)
                record = actor.feedback(event.reward, env.observation(i), terminal=env.done or ant.exhausted,
                    continuing_after_death=ant.exhausted and not env.done, tick=env.steps, individual=i) if active[i] else None
                if record != by_key.get((env.steps, i)):
                    raise AssertionError('主轨迹参数提案或接受决策重建不一致')
                if record:
                    if damage[i] > 0. and any(record.proposal.delta) and i not in selected and len(selected) < plan.pairs_per_world:
                        selected.add(i)
                        pending.append(record)
                    damage[i] = 0.
            assert env.stock == expected.food_stock and env.steps == expected.tick
            for record in pending:
                directory = output / f'seed-{seed}/tick-{env.steps:04d}-ant-{record.individual:02d}'
                evaluate(env, actors, record, horizon=plan.branch_steps, directory=directory)
                save_actors(actors, directory, env.steps, include_memory=True)
                points.append((env.steps, record.individual))
            if len(selected) >= plan.pairs_per_world or env.steps >= plan.maximum_parent_steps or env.done:
                break
    for actor in actors:
        actor.check_frozen()
    return ProbeWorld(seed=seed, parent_steps=env.steps, points=points)


def run(plan: Plan, output: Path) -> Execution:
    torch.set_num_threads(1)
    verified = verify_manifest(plan.source_root, plan.manifest)
    store = TapeStore(plan.source_root / 'on')
    if (not store.execution.plan.respawn or len(set(plan.seeds)) != len(plan.seeds)
            or not plan.seeds or not set(plan.seeds).issubset(store.execution.plan.seeds)):
        raise ValueError('需要已登记复活批次中的唯一种子')
    for source in store.execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError('源模型散列改变')
    output.mkdir(parents=True, exist_ok=False)
    execution = Execution(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        started_at=datetime.now(timezone.utc).isoformat(), plan=plan, verified_source_files=verified)
    (output / 'execution.json').write_text(execution.model_dump_json(indent=2))
    for seed in plan.seeds:
        world = probe_world(plan, store, seed, output)
        execution.worlds.append(world)
        print(world.model_dump_json(), flush=True)
    for source in store.execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError('运行后源模型改变')
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (output / 'execution.json').write_text(execution.model_dump_json(indent=2))
    return execution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(Plan.model_validate_json(args.config.read_bytes()), args.output)
