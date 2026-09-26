"""按整巢描述冻结模型基础资格，阈值筛查不作为统计等效检验。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mathhackson.training.comparison.qualification import Execution, QualificationPlan
from mathhackson.training.comparison.run import SourceRecord, WorldRecord
from mathhackson.training.comparison.auditing import Estimate, audit_world, describe, verify_manifest


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_directory: Path
    manifest: Path
    protocol: Path
    output: Path


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


def run(config: Config) -> Result:
    files = verify_manifest(config.input_directory, config.manifest)
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
    frames = sum(audit_world(directory, row, plan.environment) for row in worlds)
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
