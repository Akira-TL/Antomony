"""提前判断对照：核对真实触发时序，再计算预定生存差。"""
from __future__ import annotations

import hashlib
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ContinuousPlan, WorldResult
from .online_actor import UpdateRecord
from .return_reward_audit import audit_world, behavior_equal, frames


def audit_schedule(path: Path, world: WorldResult, plan: ContinuousPlan) -> int:
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    indexed = {(r.tick, r.individual): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('重复候选')
    pending = [0] * plan.environment.ants
    seen: set[tuple[int, int]] = set()
    for frame in frames(path):
        for i, ant in enumerate(frame.ants):
            if not ant.active or world.arm not in ('skip', 'always'):
                continue
            if ant.learning_reward is None:
                raise ValueError('缺少真实学习反馈')
            pending[i] += 1
            terminal = ant.exhausted or frame.tick == world.steps
            due = (terminal or pending[i] >= plan.adaptation.window
                   or plan.adaptation.feedback_trigger == 'negative-feedback' and ant.learning_reward < 0.)
            key = (frame.tick, i)
            if due != (key in indexed):
                raise ValueError('候选触发时序与实际反馈不符')
            if due:
                record = indexed[key]
                if record.proposal.steps != pending[i] or record.terminal != terminal:
                    raise ValueError('触发窗口长度或终止状态不符')
                seen.add(key)
                pending[i] = 0
    if seen != set(indexed) or any(pending):
        raise ValueError('候选未完整消费或记录无对应行动')
    return len(seen)


class Contrast(BaseModel):
    seed: int
    comparator: Literal['window-always', 'skip']
    failures: int
    deaths: int
    exhaustion: int
    deliveries: int


def decide(rows: list[Contrast]) -> bool:
    expected = {(s, c) for s in (19601, 19602) for c in ('window-always', 'skip')}
    if len(rows) != 4 or {(r.seed, r.comparator) for r in rows} != expected:
        raise ValueError('配对比较不完整或重复')
    return (all(r.failures <= 0 for r in rows)
            and all(any(r.failures < 0 for r in rows if r.comparator == c) for c in ('window-always', 'skip')))


class Summary(BaseModel):
    files_verified: int
    frames_verified: int
    schedules_verified: int
    controls_identical: bool
    window: list[WorldResult]
    early: list[WorldResult]
    contrasts: list[Contrast]
    continue_to_gate_design: bool


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    old, new = TapeStore(directory / 'window'), TapeStore(directory / 'early')
    expected = old.execution.plan.model_copy(update={'adaptation': old.execution.plan.adaptation.model_copy(
        update={'feedback_trigger': 'negative-feedback'})})
    if expected != new.execution.plan or old.execution.sources != new.execution.sources:
        raise ValueError('触发以外的配置或源模型发生变化')
    count = schedules = 0
    for variant, store in (('window', old), ('early', new)):
        protocol = Path(f'.research/protocols/feedback-trigger-{variant}.json').read_bytes()
        if (store.execution.plan != ContinuousPlan.model_validate_json(protocol)
                or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest()):
            raise ValueError('执行偏离冻结协议')
        for source in store.execution.sources:
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
                raise ValueError('源模型改变')
        for key, world in store.worlds.items():
            path, _ = store.world(*key)
            if world.steps != store.execution.plan.environment.horizon:
                raise ValueError('观察时限不完整')
            audit_world(path, world, store.execution.plan)
            schedules += audit_schedule(path, world, store.execution.plan)
            count += world.steps
    for key in old.worlds:
        for left, right in zip_longest(frames(old.world(*key)[0]), frames(new.world(*key)[0])):
            if (left is None or right is None or left.tick != right.tick
                    or left.source_position != right.source_position or left.source_active != right.source_active):
                raise ValueError('外部干预不一致')
            if key[2] != 'always' and not behavior_equal(left, right):
                raise ValueError('不更新的对照行为改变')
    contrasts = []
    for seed in new.execution.plan.seeds:
        focal = new.worlds[('moving-danger', seed, 'always')]
        for comparator, reference in (('window-always', old.worlds[('moving-danger', seed, 'always')]),
                                      ('skip', new.worlds[('moving-danger', seed, 'skip')])):
            contrasts.append(Contrast(seed=seed, comparator=comparator,
                failures=focal.exhausted - reference.exhausted, deaths=focal.deaths - reference.deaths,
                exhaustion=focal.exhausted - focal.deaths - reference.exhausted + reference.deaths,
                deliveries=focal.deliveries - reference.deliveries))
    return Summary(files_verified=files, frames_verified=count, schedules_verified=schedules,
        controls_identical=True, window=list(old.worlds.values()), early=list(new.worlds.values()),
        contrasts=contrasts, continue_to_gate_design=decide(contrasts))
