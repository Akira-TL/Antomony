"""返巢奖励单变量诊断：重建事件并比较完整行为，不比较异定义总分。"""
from __future__ import annotations

import gzip
import hashlib
import math
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ARMS, AntFrame, Arm, ContinuousPlan, Frame, WorldResult
from .online_actor import UpdateRecord


class Totals(BaseModel):
    pickups: int = 0
    deliveries: int = 0
    empty_returns: int = 0
    deaths: int = 0
    exhaustion: int = 0
    revivals: int = 0
    writes: int = 0
    injury: float = 0.
    reward: float = 0.
    exploration_reward: float = 0.
    return_windows: int = 0
    return_novel_windows: int = 0
    return_nonzero_windows: int = 0
    return_written_windows: int = 0


class Pair(BaseModel):
    seed: int
    arm: Arm
    on: Totals
    off: Totals
    delivery_difference: int
    death_difference: int
    exhaustion_difference: int
    empty_return_difference: int
    first_behavior_difference: int | None


class Decision(BaseModel):
    arm: Literal['learned', 'always']
    mean_delivery_difference: float
    mean_death_difference: float
    worth_further_validation: bool


class Summary(BaseModel):
    files_verified: int
    frames_verified: int
    pairs: list[Pair]
    decisions: list[Decision]
    nonlearning_controls_identical: bool


def frames(path: Path) -> Iterator[Frame]:
    with gzip.open(path / 'trajectory.jsonl.gz', 'rt') as stream:
        for line in stream:
            yield Frame.model_validate_json(line)


def behavior_equal(left: Frame, right: Frame) -> bool:
    return (left.tick == right.tick and left.source_position == right.source_position
            and left.source_active == right.source_active and left.food_stock == right.food_stock
            and len(left.ants) == len(right.ants)
            and all(a.model_copy(update={'reward': 0., 'writes': 0}) == b.model_copy(update={'reward': 0., 'writes': 0})
                    for a, b in zip(left.ants, right.ants, strict=True)))


def known_reward(ant: AntFrame, return_reward: float, injury: float,
                 deaths: int, exhaustion: int, death_cost: float, exhaustion_cost: float) -> float:
    return (float(ant.picked_up) + 4. * ant.delivered
            + return_reward * (ant.budget_return and not ant.delivered)
            - injury - death_cost * deaths - exhaustion_cost * exhaustion)


def compressed_response(raw: float) -> float:
    # 提案保存原始浓度，轨迹保存LocalObservation.vector的浮点32位压缩输入。
    return float((np.log1p(np.asarray([raw], dtype=np.float32)) / math.log(9.))[0])


def audit_world(path: Path, world: WorldResult, plan: ContinuousPlan) -> Totals:
    count = plan.environment.ants
    condition = next(c for c in plan.conditions if c.name == world.condition)
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    by_key = {(r.tick, r.individual): r for r in records}
    if len(by_key) != len(records) or len(records) != world.decisions:
        raise ValueError('提案重复或数量不符')
    pending_rewards: list[list[float]] = [[] for _ in range(count)]
    pending_returns = [0] * count
    pending_novel = [0.] * count
    result = Totals()
    previous: Frame | None = None
    consumed = 0
    for tick, frame in enumerate(frames(path), 1):
        if frame.tick != tick or len(frame.ants) != count:
            raise ValueError('轨迹时序或个体数错误')
        for i, ant in enumerate(frame.ants):
            prior = previous.ants[i] if previous else None
            deaths = ant.cumulative_deaths - (prior.cumulative_deaths if prior else 0)
            terminal = ant.cumulative_terminations - (prior.cumulative_terminations if prior else 0)
            if deaths != int(ant.active and ant.killed) or terminal != int(ant.active and ant.exhausted):
                raise ValueError('死亡或耗尽计数不符')
            injury = ant.injury - (prior.injury if prior and not ant.respawned else 0.)
            if injury < -1e-6 or ant.respawned and ant.budget_return:
                raise ValueError('伤害回退或复活误计返巢')
            known = known_reward(ant, plan.environment.return_reward, injury, deaths, terminal - deaths,
                                 condition.disturbance.death_cost, plan.environment.exhaustion_cost)
            exploration = ant.reward - known
            if not -1e-6 <= exploration <= plan.environment.exploration_reward + 1e-6:
                raise ValueError('奖励分解超出探索奖励范围')
            result.pickups += ant.picked_up
            result.deliveries += ant.delivered
            result.empty_returns += ant.budget_return and not ant.delivered
            result.deaths += deaths
            result.exhaustion += terminal - deaths
            result.revivals += ant.respawned
            result.injury += injury
            result.reward += ant.reward
            result.exploration_reward += exploration
            if world.arm in ('learned', 'always', 'skip') and ant.active:
                pending_rewards[i].append(ant.reward)
                pending_returns[i] += ant.budget_return and not ant.delivered
                pending_novel[i] = max(pending_novel[i], max(ant.observation[j] for j in range(72) if j % 8 >= 3))
            record = by_key.get((tick, i))
            if record:
                rewards = pending_rewards[i]
                if (not rewards or len(rewards) != record.proposal.steps
                        or abs(sum(rewards) / len(rewards) - record.proposal.mean_reward) > 1e-7
                        or abs(pending_novel[i] - compressed_response(record.proposal.novel_response)) > 1e-7):
                    raise ValueError('真实奖励窗口与提案不符')
                if pending_returns[i]:
                    result.return_windows += 1
                    result.return_novel_windows += pending_novel[i] > 0.
                    result.return_nonzero_windows += any(record.proposal.delta)
                    result.return_written_windows += record.changed
                result.writes += record.changed
                pending_rewards[i].clear()
                pending_returns[i] = 0
                pending_novel[i] = 0.
                consumed += 1
        if frame.food_stock is None or frame.food_stock + result.deliveries + sum(a.carrying for a in frame.ants) != plan.environment.stock:
            raise ValueError('食物库存不守恒')
        previous = frame
    if previous is None or previous.tick != world.steps or consumed != len(records) or any(pending_rewards):
        raise ValueError('轨迹或窗口未完整结束')
    if ((result.pickups, result.deliveries, result.deaths, result.exhaustion, result.revivals, result.writes)
            != (world.pickups, world.deliveries, world.deaths, world.exhausted - world.deaths,
                world.revivals, sum(world.writes))
            or abs(result.reward - world.reward) > 1e-5 or abs(result.injury - world.injury) > 1e-5):
        raise ValueError('事件重建与世界汇总不符')
    return result


def decide(pairs: list[Pair], arm: Literal['learned', 'always']) -> Decision:
    selected = [p for p in pairs if p.arm == arm]
    if len(selected) != 2 or len({p.seed for p in selected}) != 2:
        raise ValueError('判定必须含两个不同种子')
    return Decision(arm=arm,
        mean_delivery_difference=sum(p.delivery_difference for p in selected) / 2,
        mean_death_difference=sum(p.death_difference for p in selected) / 2,
        worth_further_validation=(all(p.delivery_difference >= 0 and p.death_difference <= 0 for p in selected)
            and any(p.delivery_difference > 0 or p.death_difference < 0 for p in selected)))


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    on, off = TapeStore(directory / 'on'), TapeStore(directory / 'off')
    a, b = on.execution.plan, off.execution.plan
    if (a.environment.return_reward != 2. or b.environment.return_reward != 0.
            or a.environment.model_copy(update={'return_reward': 0.}) != b.environment
            or a.model_copy(update={'environment': b.environment}) != b):
        raise ValueError('两个协议不是仅改变返巢奖励')
    for variant, store in (('on', on), ('off', off)):
        protocol = Path(f'.research/protocols/return-reward-{variant}.json').read_bytes()
        if (store.execution.plan != ContinuousPlan.model_validate_json(protocol)
                or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest()):
            raise ValueError('实际协议偏离冻结配置')
        if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in store.execution.sources):
            raise ValueError('源模型改变')
    if on.execution.sources != off.execution.sources:
        raise ValueError('配对模型来源不一致')
    pairs = []
    count = 0
    for seed in a.seeds:
        for arm in ARMS:
            left, lw = on.world('moving-danger', seed, arm)
            right, rw = off.world('moving-danger', seed, arm)
            before, after = audit_world(left, lw, a), audit_world(right, rw, b)
            first = None
            for x, y in zip_longest(frames(left), frames(right)):
                if x and y and (x.source_position != y.source_position or x.source_active != y.source_active):
                    raise ValueError('跨配置外部日程不一致')
                if first is None and (x is None or y is None or not behavior_equal(x, y)):
                    first = x.tick if x else y.tick
            if arm in ('skip', 'mlp', 'rules') and first is not None:
                raise ValueError('无更新对照行为受到奖励配置影响')
            count += lw.steps + rw.steps
            pairs.append(Pair(seed=seed, arm=arm, on=before, off=after,
                delivery_difference=after.deliveries - before.deliveries,
                death_difference=after.deaths - before.deaths,
                exhaustion_difference=after.exhaustion - before.exhaustion,
                empty_return_difference=after.empty_returns - before.empty_returns,
                first_behavior_difference=first))
    return Summary(files_verified=files, frames_verified=count, pairs=pairs,
                   decisions=[decide(pairs, arm) for arm in ('learned', 'always')],
                   nonlearning_controls_identical=True)
