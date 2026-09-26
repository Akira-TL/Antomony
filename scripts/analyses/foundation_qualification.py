"""按整巢描述冻结模型基础资格，阈值筛查不作为统计等效检验。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from mathhackson.training.comparison.qualification import Execution, QualificationPlan
from mathhackson.training.comparison.run import Frame, SourceRecord, WorldRecord


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_directory: Path
    manifest: Path
    protocol: Path
    output: Path


class Estimate(BaseModel):
    mean: float
    minimum: float
    maximum: float


def describe(values: list[float]) -> Estimate:
    if not values or not np.isfinite(values).all():
        raise ValueError("描述数据为空或非有限")
    return Estimate(mean=float(np.mean(values)), minimum=min(values), maximum=max(values))


class Group(BaseModel):
    model: str
    delivery_fraction: Estimate
    complete_food_worlds: int
    completion_steps: Estimate | None
    empty_returning_fraction: Estimate
    empty_exhausted_fraction: Estimate
    empty_max_radius: Estimate
    delivery_ready: bool
    return_ready: bool
    exploration_ready: bool
    minimum_ready: bool


class Contrast(BaseModel):
    comparator: str
    delivery_fraction: Estimate
    paired_completion_steps: Estimate | None
    complete_pairs: int
    empty_returning_fraction: Estimate
    empty_exhausted_fraction: Estimate
    empty_max_radius: Estimate


class Result(BaseModel):
    files_verified: int
    frames_verified: int
    execution: Execution
    worlds: list[WorldRecord]
    groups: list[Group]
    recurrent_minus_comparator: list[Contrast]
    all_minimum_ready: bool
    equivalence_established: bool = False


def verify_manifest(config: Config) -> int:
    seen: set[Path] = set()
    for line in config.manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if path in seen or not path.resolve().is_relative_to(config.input_directory.resolve()):
            raise ValueError("重复或越界的文件")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"文件散列改变：{path}")
        seen.add(path)
    if seen != {p for p in config.input_directory.rglob("*") if p.is_file()}:
        raise ValueError("清单不完整")
    return len(seen)


def audit_world(directory: Path, row: WorldRecord, plan: QualificationPlan) -> int:
    count = plan.environment.ants
    with gzip.open(directory / f"{row.model}-{row.task}-{row.seed}-{row.sampled}.jsonl.gz", "rt") as stream:
        frames = [Frame.model_validate_json(line) for line in stream]
    if len(frames) != row.steps or not frames or row.steps > plan.environment.horizon:
        raise ValueError("世界步数或轨迹长度错误")
    if row.updates != [0] * count or row.sampled != (None if row.model == "rules" else True):
        raise ValueError("发生训练或采样模式错误")
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


def run(config: Config) -> Result:
    files = verify_manifest(config)
    directory = config.input_directory
    plan = QualificationPlan.model_validate_json((directory / "config.json").read_text())
    expected = QualificationPlan.model_validate_json(config.protocol.read_text())
    if plan != expected:
        raise ValueError("实际与冻结协议不同")
    execution = Execution.model_validate_json((directory / "execution.json").read_text())
    if execution.smoke or execution.training_updates or not execution.completed_at:
        raise ValueError("采样未完成或包含训练/短流程")
    sources = [SourceRecord.model_validate_json(line) for line in (directory / "sources.jsonl").read_text().splitlines()]
    if len(sources) != 1 + 2 * plan.environment.ants or len({s.path for s in sources}) != len(sources):
        raise ValueError("来源数量不符")
    for source in sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError("源模型文件已改变")
    worlds = [WorldRecord.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
    identities = {(seed, task, model) for seed in plan.seeds for task in ("food", "empty") for model in ("mlp", "recurrent", "rules")}
    if len(worlds) != len(identities) or {(r.seed, r.task, r.model) for r in worlds} != identities:
        raise ValueError("世界重复、缺失或多余")
    frames = sum(audit_world(directory, row, plan) for row in worlds)
    groups: list[Group] = []
    for name in ("mlp", "recurrent", "rules"):
        food = [r for r in worlds if r.model == name and r.task == "food"]
        empty = [r for r in worlds if r.model == name and r.task == "empty"]
        complete = [float(r.steps) for r in food if r.deliveries == plan.environment.stock]
        delivery = describe([r.deliveries / plan.environment.stock for r in food])
        returning = describe([r.returning_individuals / plan.environment.ants for r in empty])
        radius = describe([r.mean_max_radius for r in empty])
        ready = (delivery.mean >= plan.minimum_delivery_fraction,
                 returning.mean >= plan.minimum_returning_fraction,
                 radius.mean >= plan.minimum_exploration_radius)
        groups.append(Group(model=name, delivery_fraction=delivery, complete_food_worlds=len(complete),
            completion_steps=describe(complete) if complete else None, empty_returning_fraction=returning,
            empty_exhausted_fraction=describe([r.exhausted / plan.environment.ants for r in empty]),
            empty_max_radius=radius, delivery_ready=ready[0], return_ready=ready[1], exploration_ready=ready[2], minimum_ready=all(ready)))
    by_id = {(r.seed, r.task, r.model): r for r in worlds}
    contrasts = []
    for model in ("mlp", "rules"):
        food = [(by_id[seed, "food", "recurrent"], by_id[seed, "food", model]) for seed in plan.seeds]
        empty = [(by_id[seed, "empty", "recurrent"], by_id[seed, "empty", model]) for seed in plan.seeds]
        completed = [float(a.steps - b.steps) for a, b in food if a.deliveries == b.deliveries == plan.environment.stock]
        contrasts.append(Contrast(comparator=model,
            delivery_fraction=describe([(a.deliveries - b.deliveries) / plan.environment.stock for a, b in food]),
            paired_completion_steps=describe(completed) if completed else None, complete_pairs=len(completed),
            empty_returning_fraction=describe([(a.returning_individuals - b.returning_individuals) / plan.environment.ants for a, b in empty]),
            empty_exhausted_fraction=describe([(a.exhausted - b.exhausted) / plan.environment.ants for a, b in empty]),
            empty_max_radius=describe([a.mean_max_radius - b.mean_max_radius for a, b in empty])))
    result = Result(files_verified=files, frames_verified=frames, execution=execution, worlds=worlds,
                    groups=groups, recurrent_minus_comparator=contrasts, all_minimum_ready=all(g.minimum_ready for g in groups))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(Config.model_validate_json(args.config.read_text()))
    print(result.model_dump_json(include={"files_verified", "frames_verified", "groups", "recurrent_minus_comparator", "all_minimum_ready"}, indent=2))
