"""同状态伤害候选的描述分析，核对真实分支事件而不产生在线训练标签。"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import Frame
from .injury_branch import BranchPair, Outcome
from .injury_probe import Execution


class Row(BaseModel):
    seed: int
    tick: int
    individual: int
    death_window: bool
    originally_accepted: bool
    prediction: float
    reward_difference: float
    delivery_difference: int
    death_difference: int
    exhaustion_difference: int
    injury_difference: float
    colony_delivery_difference: int
    colony_death_difference: int


class Group(BaseModel):
    seed: int
    points: int
    improved_accepted: int
    improved_rejected: int
    worse_accepted: int
    worse_rejected: int
    tied_accepted: int
    tied_rejected: int
    death_saving_rejected: int
    delivery_gaining_rejected: int


class Result(BaseModel):
    files_verified: int
    frames_verified: int
    execution: Execution
    rows: list[Row]
    groups: list[Group]
    missed_benefit_in_both_worlds: bool


def describe(seed: int, rows: list[Row]) -> Group:
    rows = [r for r in rows if r.seed == seed]
    return Group(seed=seed, points=len(rows),
        improved_accepted=sum(r.reward_difference > 1e-6 and r.originally_accepted for r in rows),
        improved_rejected=sum(r.reward_difference > 1e-6 and not r.originally_accepted for r in rows),
        worse_accepted=sum(r.reward_difference < -1e-6 and r.originally_accepted for r in rows),
        worse_rejected=sum(r.reward_difference < -1e-6 and not r.originally_accepted for r in rows),
        tied_accepted=sum(abs(r.reward_difference) <= 1e-6 and r.originally_accepted for r in rows),
        tied_rejected=sum(abs(r.reward_difference) <= 1e-6 and not r.originally_accepted for r in rows),
        death_saving_rejected=sum(r.death_difference < 0 and not r.originally_accepted for r in rows),
        delivery_gaining_rejected=sum(r.delivery_difference > 0 and not r.originally_accepted for r in rows))


def audit_branch(path: Path, outcome: Outcome, parent: Frame, focal: int, stock: int,
                 branch_steps: int, world_horizon: int) -> int:
    previous = parent
    injury = reward = 0.
    delivered = colony_delivered = revived = count = 0
    initial_deliveries = stock - parent.food_stock - sum(a.carrying for a in parent.ants)
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            frame = Frame.model_validate_json(line)
            if frame.tick != previous.tick + 1 or len(frame.ants) != len(parent.ants):
                raise ValueError('分支时间或个体身份错误')
            ant, prior = frame.ants[focal], previous.ants[focal]
            injury += ant.injury - (0. if ant.respawned else prior.injury)
            reward += ant.reward
            delivered += ant.delivered
            colony_delivered += sum(a.delivered for a in frame.ants)
            revived += ant.respawned
            if (frame.food_stock + initial_deliveries + colony_delivered + sum(a.carrying for a in frame.ants) != stock
                    or any(a.writes for a in frame.ants)):
                raise ValueError('分支食物不守恒或发生额外参数更新')
            previous = frame
            count += 1
    ant, prior = previous.ants[focal], parent.ants[focal]
    death = ant.cumulative_deaths - prior.cumulative_deaths
    exhaustion = ant.cumulative_terminations - prior.cumulative_terminations - death
    colony_deaths = sum(a.cumulative_deaths - b.cumulative_deaths for a, b in zip(previous.ants, parent.ants, strict=True))
    actual = Outcome(steps=count, focal_reward=reward, focal_deliveries=delivered, focal_deaths=death,
        focal_exhaustion=exhaustion, focal_injury=injury, focal_revivals=revived,
        colony_deliveries=colony_delivered, colony_deaths=colony_deaths)
    if not 1 <= count <= branch_steps or (count < branch_steps
            and previous.tick < world_horizon and initial_deliveries + colony_delivered < stock):
        raise ValueError('分支提前结束却没有达到世界终止条件')
    if actual.model_copy(update={'focal_reward': outcome.focal_reward, 'focal_injury': outcome.focal_injury}) != outcome:
        raise ValueError('分支事件与结果不符')
    if abs(actual.focal_reward - outcome.focal_reward) > 1e-7 or abs(actual.focal_injury - outcome.focal_injury) > 1e-6:
        raise ValueError('分支奖励或伤害不符')
    return count


def audit(directory: Path, manifest: Path) -> Result:
    files = verify_manifest(directory, manifest)
    execution = Execution.model_validate_json((directory / 'execution.json').read_bytes())
    if not execution.completed_at or tuple(w.seed for w in execution.worlds) != execution.plan.seeds:
        raise ValueError('采样未完整完成')
    store = TapeStore(execution.plan.source_root / 'on')
    verify_manifest(execution.plan.source_root, execution.plan.manifest)
    rows: list[Row] = []
    count = 0
    for world in execution.worlds:
        if len(world.points) > execution.plan.pairs_per_world or len({i for _, i in world.points}) != len(world.points):
            raise ValueError('采样超预算或重复个体')
        source, _ = store.world('moving-danger', world.seed, 'learned')
        ticks = {tick for tick, _ in world.points}
        parents: dict[int, Frame] = {}
        with gzip.open(source / 'trajectory.jsonl.gz', 'rt') as stream:
            for line in stream:
                frame = Frame.model_validate_json(line)
                if frame.tick in ticks:
                    parents[frame.tick] = frame
                if frame.tick >= world.parent_steps:
                    break
        for tick, individual in world.points:
            point = directory / f'seed-{world.seed}/tick-{tick:04d}-ant-{individual:02d}'
            pair = BranchPair.model_validate_json((point / 'pair.json').read_bytes())
            record = pair.parent_choice
            if (pair.tick, pair.individual) != (tick, individual) or not pair.changed or not record.eligible:
                raise ValueError('采样身份或候选有效性不符')
            if pair.accepted_weights == record.before:
                raise ValueError('候选未改变')
            for i in range(store.execution.plan.environment.ants):
                with np.load(point / f'tick-{tick:04d}-ant-{i:02d}.residual.npz') as saved:
                    if i == individual:
                        np.testing.assert_array_equal(saved['stable'] + saved['fast'], record.after)
                with np.load(point / f'tick-{tick:04d}-ant-{i:02d}.memory.npz') as saved:
                    if saved['history'].shape[1:] != (8,) or saved['random_state'].dtype != np.uint8:
                        raise ValueError('父状态快照非法')
            for mode, outcome in (('skip', pair.skip), ('accept', pair.accept)):
                count += audit_branch(point / f'{mode}.jsonl.gz', outcome, parents[tick], individual,
                    store.execution.plan.environment.stock, execution.plan.branch_steps, store.execution.plan.environment.horizon)
            a, s = pair.accept, pair.skip
            rows.append(Row(seed=world.seed, tick=tick, individual=individual, death_window=record.terminal,
                originally_accepted=record.accepted, prediction=record.prediction,
                reward_difference=a.focal_reward - s.focal_reward,
                delivery_difference=a.focal_deliveries - s.focal_deliveries,
                death_difference=a.focal_deaths - s.focal_deaths,
                exhaustion_difference=a.focal_exhaustion - s.focal_exhaustion,
                injury_difference=a.focal_injury - s.focal_injury,
                colony_delivery_difference=a.colony_deliveries - s.colony_deliveries,
                colony_death_difference=a.colony_deaths - s.colony_deaths))
    groups = [describe(seed, rows) for seed in execution.plan.seeds]
    return Result(files_verified=files, frames_verified=count, execution=execution, rows=rows, groups=groups,
                  missed_benefit_in_both_worlds=len(groups) == 2 and all(g.improved_rejected > 0 for g in groups))
