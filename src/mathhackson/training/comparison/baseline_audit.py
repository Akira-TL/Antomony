"""重建历史基线并核对单变量连续对照；不调参或生成训练标签。"""
from __future__ import annotations

import hashlib
from itertools import zip_longest
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ARMS, Arm, ContinuousPlan, WorldResult
from .online_actor import UpdateRecord
from .return_reward_audit import Totals, audit_world, behavior_equal, frames


class History(BaseModel):
    seed: int
    arm: Arm
    rate: float
    snapshots: int
    windows: int
    minimum: float
    maximum: float


class Pair(BaseModel):
    seed: int
    arm: Arm
    disabled: Totals
    enabled: Totals
    delivery_difference: int
    death_difference: int
    exhaustion_difference: int
    reward_difference: float
    write_difference: int


class Summary(BaseModel):
    files_verified: int
    frames_verified: int
    pairs: list[Pair]
    histories: list[History]
    nonlearning_controls_identical: bool
    worth_further_validation: bool


def next_baseline(previous: float, rewards: list[float], gamma: float, rate: float) -> float:
    if not rewards:
        raise ValueError("不能由空窗口重建基线")
    returns = np.asarray([sum(gamma ** j * r for j, r in enumerate(rewards[i:]))
                          for i in range(len(rewards))], dtype=np.float32)
    return (1. - rate) * previous + rate * float(returns.mean())


def history(path: Path, world: WorldResult, plan: ContinuousPlan) -> History:
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    by_key = {(r.tick, r.individual): r for r in records}
    n, rate = plan.environment.ants, plan.adaptation.historical_baseline_rate
    pending: list[list[float]] = [[] for _ in range(n)]
    baselines, counts = [0.] * n, [0] * n
    low = high = 0.
    snapshots = 0

    def check(tick: int) -> None:
        nonlocal snapshots
        for i in range(n):
            with np.load(path / f'tick-{tick:04d}-ant-{i:02d}.residual.npz', allow_pickle=False) as saved:
                if (abs(float(saved['return_baseline']) - baselines[i]) > 1e-6
                        or int(saved['baseline_windows']) != counts[i]):
                    raise ValueError('基线快照与过去已处理反馈不符')
            snapshots += 1

    check(0)
    for frame in frames(path):
        for i, ant in enumerate(frame.ants):
            if ant.active:
                pending[i].append(ant.reward)
            record = by_key.get((frame.tick, i))
            if record:
                if len(pending[i]) != record.proposal.steps:
                    raise ValueError('反馈窗口长度错误')
                if rate:
                    baselines[i] = next_baseline(baselines[i], pending[i], plan.adaptation.gamma, rate)
                    counts[i] += 1
                    low, high = min(low, baselines[i]), max(high, baselines[i])
                pending[i].clear()
        if frame.tick in world.snapshots:
            check(frame.tick)
    if any(pending):
        raise ValueError('仍有未处理的真实反馈')
    return History(seed=world.seed, arm=world.arm, rate=rate, snapshots=snapshots,
                   windows=sum(counts), minimum=low, maximum=high)


def worth_further_validation(pairs: list[Pair]) -> bool:
    chosen = [p for p in pairs if p.arm == 'always']
    if len(chosen) != 2 or len({p.seed for p in chosen}) != 2:
        raise ValueError('必须有两个不同种子的固定接受配对')
    return (all(p.delivery_difference >= 0 and p.death_difference <= 0
                and p.exhaustion_difference <= 0 for p in chosen)
            and any(p.delivery_difference > 0 or p.death_difference < 0 for p in chosen))


def audit(directory: Path, manifest: Path) -> Summary:
    verified = verify_manifest(directory, manifest)
    off, on = TapeStore(directory / 'off'), TapeStore(directory / 'on')
    a, b = off.execution.plan, on.execution.plan
    if (a.adaptation.historical_baseline_rate != 0. or b.adaptation.historical_baseline_rate != .1
            or b.adaptation.model_copy(update={'historical_baseline_rate': 0.}) != a.adaptation
            or b.model_copy(update={'adaptation': a.adaptation}) != a):
        raise ValueError('配置不是仅改变预定基线速率')
    for variant, store in (('off', off), ('on', on)):
        protocol = Path(f'.research/protocols/historical-baseline-{variant}.json').read_bytes()
        if (store.execution.plan != ContinuousPlan.model_validate_json(protocol)
                or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest()):
            raise ValueError('执行偏离冻结配置')
        if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in store.execution.sources):
            raise ValueError('源参数改变')
    if off.execution.sources != on.execution.sources:
        raise ValueError('两配置来源不同')
    pairs: list[Pair] = []
    histories: list[History] = []
    count = 0
    for seed in a.seeds:
        for arm in ARMS:
            left, lw = off.world('moving-danger', seed, arm)
            right, rw = on.world('moving-danger', seed, arm)
            x, y = audit_world(left, lw, a), audit_world(right, rw, b)
            if arm in ('learned', 'skip', 'always'):
                histories.extend((history(left, lw, a), history(right, rw, b)))
            for before, after in zip_longest(frames(left), frames(right)):
                if before and after and (before.source_position != after.source_position
                                         or before.source_active != after.source_active):
                    raise ValueError('外部干预未同步')
                if arm in ('skip', 'mlp', 'rules') and (before is None or after is None or not behavior_equal(before, after)):
                    raise ValueError('无更新对照的行为改变')
            count += lw.steps + rw.steps
            pairs.append(Pair(seed=seed, arm=arm, disabled=x, enabled=y,
                delivery_difference=y.deliveries - x.deliveries, death_difference=y.deaths - x.deaths,
                exhaustion_difference=y.exhaustion - x.exhaustion, reward_difference=y.reward - x.reward,
                write_difference=y.writes - x.writes))
    return Summary(files_verified=verified, frames_verified=count, pairs=pairs, histories=histories,
        nonlearning_controls_identical=True, worth_further_validation=worth_further_validation(pairs))
