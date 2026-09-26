"""固定生存反馈设计的逐帧核验及世界配对差，不用旧总分判断生存。"""
from __future__ import annotations

import hashlib
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ContinuousPlan, WorldResult
from .return_reward_audit import audit_world, behavior_equal, frames


class Contrast(BaseModel):
    seed: int
    comparator: Literal['legacy-always', 'skip']
    failures: int
    deaths: int
    exhaustion: int
    deliveries: int
    injury: float


class Summary(BaseModel):
    files_verified: int
    frames_verified: int
    controls_identical: bool
    legacy: list[WorldResult]
    survival: list[WorldResult]
    contrasts: list[Contrast]
    continue_to_gate_design: bool


def contrast(new: WorldResult, old: WorldResult, comparator: Literal['legacy-always', 'skip']) -> Contrast:
    if new.seed != old.seed or new.steps != old.steps:
        raise ValueError('配对身份或观察时限不一致')
    return Contrast(seed=new.seed, comparator=comparator,
        failures=new.exhausted - old.exhausted, deaths=new.deaths - old.deaths,
        exhaustion=(new.exhausted - new.deaths) - (old.exhausted - old.deaths),
        deliveries=new.deliveries - old.deliveries, injury=new.injury - old.injury)


def decide(rows: list[Contrast]) -> bool:
    expected = {(seed, comparator) for seed in (19501, 19502) for comparator in ('legacy-always', 'skip')}
    if len(rows) != 4 or {(r.seed, r.comparator) for r in rows} != expected:
        raise ValueError('固定比较缺失或重复')
    return (all(r.failures <= 0 for r in rows)
            and all(any(r.failures < 0 for r in rows if r.comparator == c) for c in ('legacy-always', 'skip')))


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    old, new = TapeStore(directory / 'legacy'), TapeStore(directory / 'survival')
    if old.execution.plan.model_copy(update={'feedback_profile': 'survival-v1'}) != new.execution.plan:
        raise ValueError('反馈以外的配置发生变化')
    if old.execution.sources != new.execution.sources:
        raise ValueError('来源模型不一致')
    count = 0
    for variant, store in (('legacy', old), ('survival', new)):
        protocol = Path(f'.research/protocols/survival-feedback-{variant}.json').read_bytes()
        plan = store.execution.plan
        if plan != ContinuousPlan.model_validate_json(protocol) or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest():
            raise ValueError('偏离冻结配置')
        for source in store.execution.sources:
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
                raise ValueError('源模型改变')
        for key, world in store.worlds.items():
            if world.steps != plan.environment.horizon:
                raise ValueError('观察时限不完整')
            audit_world(store.world(*key)[0], world, plan)
            count += world.steps
    for key in old.worlds:
        for left, right in zip_longest(frames(old.world(*key)[0]), frames(new.world(*key)[0])):
            if left is None or right is None or left.tick != right.tick or left.source_position != right.source_position or left.source_active != right.source_active:
                raise ValueError('外部日程不同步')
            if key[2] != 'always' and not behavior_equal(left, right):
                raise ValueError('不更新的对照行为改变')
    rows = []
    for seed in new.execution.plan.seeds:
        focal = new.worlds[('moving-danger', seed, 'always')]
        rows.append(contrast(focal, old.worlds[('moving-danger', seed, 'always')], 'legacy-always'))
        rows.append(contrast(focal, new.worlds[('moving-danger', seed, 'skip')], 'skip'))
    return Summary(files_verified=files, frames_verified=count, controls_identical=True,
        legacy=list(old.worlds.values()), survival=list(new.worlds.values()), contrasts=rows,
        continue_to_gate_design=decide(rows))
