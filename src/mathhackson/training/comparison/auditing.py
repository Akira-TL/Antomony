"""基础及课程评价共用的原始轨迹核对，不产生策略效果结论。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mathhackson.training.foraging.colony import ColonyConfig
from .run import Frame, WorldRecord


class Estimate(BaseModel):
    mean: float
    minimum: float
    maximum: float


def describe(values: list[float]) -> Estimate:
    if not values or not np.isfinite(values).all():
        raise ValueError("描述数据为空或非有限")
    return Estimate(mean=float(np.mean(values)), minimum=min(values), maximum=max(values))


def verify_manifest(directory: Path, manifest: Path) -> int:
    seen: set[Path] = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if path in seen or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("重复或越界的文件")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"文件散列改变：{path}")
        seen.add(path)
    if seen != {p for p in directory.rglob("*") if p.is_file()}:
        raise ValueError("清单不完整")
    return len(seen)


def audit_world(directory: Path, row: WorldRecord, environment: ColonyConfig, *, frozen: bool = True) -> int:
    count = environment.ants
    with gzip.open(directory / f"{row.model}-{row.task}-{row.seed}-{row.sampled}.jsonl.gz", "rt") as stream:
        frames = [Frame.model_validate_json(line) for line in stream]
    if len(frames) != row.steps or not frames or row.steps > environment.horizon:
        raise ValueError("世界步数或轨迹长度错误")
    if row.sampled != (None if row.model == "rules" else True) or (frozen and row.updates != [0] * count):
        raise ValueError("发生训练或采样模式错误")
    if len(row.updates) != count or any(v < 0 for v in row.updates):
        raise ValueError("更新计数错误")
    carrying = np.zeros(count, dtype=np.bool_)
    radius = np.zeros(count)
    pickups = deliveries = 0
    for tick, frame in enumerate(frames, 1):
        if frame.tick != tick or any(len(v) != count for v in (
                frame.positions, frame.moves, frame.turns, frame.carrying, frame.rewards)):
            raise ValueError("轨迹时序或个体数错误")
        positions = np.asarray(frame.positions, dtype=np.float32)
        if positions.shape != (count, 2) or not np.isfinite(positions).all() or not np.isfinite(frame.rewards).all():
            raise ValueError("非法轨迹数值")
        current = np.asarray(frame.carrying)
        pickups += int((current & ~carrying).sum())
        deliveries += int((carrying & ~current).sum())
        carrying = current
        radius = np.maximum(radius, np.linalg.norm(positions, axis=1))
    if (pickups, deliveries) != (row.pickups, row.deliveries):
        raise ValueError("携食变更与交付记录不符")
    if not np.isclose(float(radius.mean()), row.mean_max_radius, atol=1e-5, rtol=0.):
        raise ValueError("探索半径与轨迹不符")
    if row.task == "empty" and (pickups or deliveries):
        raise ValueError("无食物世界发生食物交互")
    return len(frames)
