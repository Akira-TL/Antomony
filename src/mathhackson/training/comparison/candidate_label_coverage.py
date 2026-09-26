"""只读既有候选后果，枚举接受或跳过可达到的固定集合交付上界。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.foraging.update_curriculum import read_partition, world_key
from .acceptance_audit import Audit, audit_inputs, audit_pairs


class Identity(BaseModel):
    path: Path
    sha256: str


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    input_directory: Path
    manifest: Path
    protocol: Path
    identities: list[Identity]
    output: Path


class Label(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    partition: Literal['train', 'held']
    world: str
    seed: int
    initial_norm: float
    tick: int
    individual: int
    skip_deliveries: int
    accept_deliveries: int
    delivery_difference: int
    skip_reward: float
    accept_reward: float
    reward_difference: float


class Counts(BaseModel):
    positive: int
    negative: int
    zero: int


class Group(BaseModel):
    partition: str
    seed: int | None = None
    initial_norm: float | None = None
    individual: int | None = None
    count: int
    deliveries: Counts
    rewards: Counts
    maximum_delivery_sum: int
    maximum_delivery_mean: float | None
    always_delivery_sum: int


def signs(values: list[float], tolerance: float) -> Counts:
    x = np.asarray(values, np.float64)
    if tolerance < 0 or not np.isfinite(x).all():
        raise ValueError('符号计数只接受有限数值和非负容差')
    return Counts(positive=int((x > tolerance).sum()), negative=int((x < -tolerance).sum()),
                  zero=int((np.abs(x) <= tolerance).sum()))


def summarize(rows: list[Label], partition: str, *, seed: int | None = None,
              initial_norm: float | None = None, individual: int | None = None) -> Group:
    selected = [row for row in rows if row.partition == partition
                and (seed is None or row.seed == seed)
                and (initial_norm is None or row.initial_norm == initial_norm)
                and (individual is None or row.individual == individual)]
    difference = [row.delivery_difference for row in selected]
    maximum = sum(max(value, 0) for value in difference)
    return Group(partition=partition, seed=seed, initial_norm=initial_norm, individual=individual,
        count=len(selected), deliveries=signs(difference, 0.),
        rewards=signs([row.reward_difference for row in selected], 1e-6),
        maximum_delivery_sum=maximum, maximum_delivery_mean=maximum / len(selected) if selected else None,
        always_delivery_sum=sum(difference))


class Result(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str
    elapsed_seconds: float
    audit: Audit
    labels: list[Label]
    partitions: list[Group]
    seeds: list[Group]
    worlds: list[Group]
    individuals: list[Group]
    world_individuals: list[Group]


def run(config: Config) -> Result:
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    for identity in config.identities:
        if hashlib.sha256(identity.path.read_bytes()).hexdigest() != identity.sha256:
            raise ValueError('已固定的输入身份改变')
    execution, audit = audit_inputs(config.input_directory, config.manifest, config.protocol)
    plan = execution.plan
    if (plan.train_seeds != tuple(range(16101, 16109)) or plan.held_seeds != tuple(range(16111, 16115))
            or plan.initial_norms != (0., .75) or plan.probe.environment.ants != 8):
        raise ValueError('不是获准的既有课程')
    labels: list[Label] = []
    for partition, expected in (('train', 196), ('held', 105)):
        rows = read_partition(config.input_directory, plan, partition)
        audit_pairs(rows, plan)
        if len(rows) != expected:
            raise ValueError('候选数量与已登记课程不一致')
        for row in rows:
            record = row.record
            norm = next(norm for index, norm in enumerate(plan.initial_norms)
                        if row.key == world_key(index, record.seed))
            a, b = record.result.accept, record.result.skip
            if (min(a.focal_deliveries, b.focal_deliveries) < 0
                    or max(a.focal_deliveries, b.focal_deliveries) > plan.probe.environment.stock):
                raise ValueError('焦点交付计数越界')
            labels.append(Label(partition=partition, world=row.key, seed=record.seed, initial_norm=norm,
                tick=record.tick, individual=record.focal, skip_deliveries=b.focal_deliveries,
                accept_deliveries=a.focal_deliveries, delivery_difference=a.focal_deliveries - b.focal_deliveries,
                skip_reward=b.focal_reward, accept_reward=a.focal_reward,
                reward_difference=a.focal_reward - b.focal_reward))
    worlds, seeds, individuals, world_individuals = [], [], [], []
    for partition, seed_values in (('train', plan.train_seeds), ('held', plan.held_seeds)):
        individuals.extend(summarize(labels, partition, individual=i) for i in range(8))
        for seed in seed_values:
            seeds.append(summarize(labels, partition, seed=seed))
            for norm in plan.initial_norms:
                worlds.append(summarize(labels, partition, seed=seed, initial_norm=norm))
                world_individuals.extend(summarize(labels, partition, seed=seed, initial_norm=norm, individual=i)
                                         for i in range(8))
    result = Result(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        started_at=started, completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
        audit=audit, labels=labels, partitions=[summarize(labels, partition) for partition in ('train', 'held')],
        seeds=seeds, worlds=worlds, individuals=individuals, world_individuals=world_individuals)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open('x') as stream:
        stream.write(result.model_dump_json(indent=2))
    return result
