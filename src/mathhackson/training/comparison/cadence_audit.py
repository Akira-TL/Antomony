"""固定四步与十六步窗口的配对核验与生存差值。"""
from __future__ import annotations

import hashlib
from itertools import zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import Arm, ContinuousPlan, WorldResult
from .return_reward_audit import audit_world, behavior_equal, frames
from .trigger_audit import audit_schedule


class Contrast(BaseModel):
    seed: int
    comparator: Literal['sixteen-always', 'skip']
    failures: int
    deaths: int
    exhaustion: int
    deliveries: int


class Activity(BaseModel):
    variant: Literal['sixteen', 'four']
    seed: int
    arm: Arm
    moves: int


class Summary(BaseModel):
    files_verified: int
    frames_verified: int
    schedules_verified: int
    controls_identical: bool
    sixteen: list[WorldResult]
    four: list[WorldResult]
    activity: list[Activity]
    contrasts: list[Contrast]
    continue_to_gate_design: bool


def validate_pair(sixteen: ContinuousPlan, four: ContinuousPlan) -> None:
    if (sixteen.adaptation.window != 16 or four.adaptation.window != 4
            or sixteen.adaptation.feedback_trigger != 'window'
            or sixteen.feedback_profile != 'survival-v1'
            or sixteen.seeds != (19701, 19702)
            or sixteen.arms != ('skip', 'always', 'mlp', 'rules')
            or sixteen.model_copy(update={'adaptation': sixteen.adaptation.model_copy(update={'window': 4})}) != four):
        raise ValueError('配对配置必须仅改变固定窗口16到4')


def decide(rows: list[Contrast]) -> bool:
    expected = {(s, c) for s in (19701, 19702) for c in ('sixteen-always', 'skip')}
    if len(rows) != 4 or {(r.seed, r.comparator) for r in rows} != expected:
        raise ValueError('配对比较不完整或重复')
    if any(r.failures != r.deaths + r.exhaustion for r in rows):
        raise ValueError('失败差与死亡耗尽分量不符')
    return (all(r.failures <= 0 for r in rows)
            and all(any(r.failures < 0 for r in rows if r.comparator == c) for c in ('sixteen-always', 'skip')))


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    old, new = TapeStore(directory / 'sixteen'), TapeStore(directory / 'four')
    validate_pair(old.execution.plan, new.execution.plan)
    if old.execution.sources != new.execution.sources:
        raise ValueError('配对模型来源不一致')
    count = schedules = 0
    activity = []
    for variant, store, name in (('sixteen', old, 'window-cadence-16.json'),
                                 ('four', new, 'four-frame-development.json')):
        protocol = (Path('.research/protocols') / name).read_bytes()
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
            activity.append(Activity(variant=variant, seed=world.seed, arm=world.arm,
                moves=sum(a.active and a.move for frame in frames(path) for a in frame.ants)))
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
        for comparator, reference in (('sixteen-always', old.worlds[('moving-danger', seed, 'always')]),
                                      ('skip', new.worlds[('moving-danger', seed, 'skip')])):
            contrasts.append(Contrast(seed=seed, comparator=comparator,
                failures=focal.exhausted - reference.exhausted, deaths=focal.deaths - reference.deaths,
                exhaustion=focal.exhausted - focal.deaths - reference.exhausted + reference.deaths,
                deliveries=focal.deliveries - reference.deliveries))
    return Summary(files_verified=files, frames_verified=count, schedules_verified=schedules,
        controls_identical=True, sixteen=list(old.worlds.values()), four=list(new.worlds.values()),
        activity=activity, contrasts=contrasts, continue_to_gate_design=decide(contrasts))
