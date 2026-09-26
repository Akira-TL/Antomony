"""固定最终课程模型的配对评价，保留无改善与能力损失。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.comparison.auditing import Estimate, audit_world, describe, verify_manifest
from mathhackson.training.comparison.exploration_course import CoursePlan, Execution
from mathhackson.training.comparison.run import SourceRecord, WorldRecord
from mathhackson.training.foraging.policy import ForagingPolicy


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
    returning_fraction: Estimate
    exhausted_fraction: Estimate
    exploration_radius: Estimate


class ParameterChange(BaseModel):
    ant: int
    final_updates: int
    change_norm: float


class Contrast(BaseModel):
    delivery_fraction: Estimate
    returning_fraction: Estimate
    exhausted_fraction: Estimate
    exploration_radius: Estimate
    completion_steps: Estimate | None
    complete_pairs: int


class Result(BaseModel):
    files_verified: int
    frames_verified: int
    snapshots_verified: int
    execution: Execution
    worlds: list[WorldRecord]
    groups: list[Group]
    after_minus_before: Contrast
    parameter_changes: list[ParameterChange]
    retain_food: bool
    retain_return: bool
    exploration_ready: bool
    adopt_final: bool


def snapshots(directory: Path, plan: CoursePlan, execution: Execution, training: list[WorldRecord]) -> list[ParameterChange]:
    episodes = [0, *range(plan.checkpoint_every, len(plan.training_seeds) + 1, plan.checkpoint_every)]
    if episodes[-1] != len(plan.training_seeds):
        episodes.append(len(plan.training_seeds))
    if execution.snapshot_episodes != episodes or execution.final_updates != training[-1].updates:
        raise ValueError("快照时间点或最终更新数不符")
    root = directory / "snapshots"
    expected = {root / f"episode-{episode:04d}-ant-{ant:02d}{suffix}"
                for episode in episodes for ant in range(plan.environment.ants) for suffix in (".npz", ".optimizer.pt")}
    if set(root.iterdir()) != expected:
        raise ValueError("参数或优化器快照不完整")
    changes = []
    for ant in range(plan.environment.ants):
        original = ForagingPolicy.load(Path(plan.recurrent_directory) / f"episode-0008-ant-{ant:02d}.npz")
        for episode in episodes:
            path = root / f"episode-{episode:04d}-ant-{ant:02d}.npz"
            model = ForagingPolicy.load(path)
            update = training[episode - 1].updates[ant] if episode else 0
            with np.load(path, allow_pickle=False) as data:
                if int(data["update"]) != update:
                    raise ValueError("快照更新数不一致")
            if any(bool(p.any()) for p in model.reserved_parameters()):
                raise ValueError("预留接收器被训练")
            if episode == 0 and any(not a.equal(b) for a, b in zip(model.parameters(), original.parameters(), strict=True)):
                raise ValueError("初始参数不是登记来源")
            optimizer = torch.load(path.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=True)
            if any(g["lr"] != .0003 or g["weight_decay"] != .0001 for g in optimizer["param_groups"]):
                raise ValueError("优化器设置改变")
            steps = [int(state["step"]) for state in optimizer["state"].values()]
            if max(steps, default=0) != update:
                raise ValueError("优化器状态与参数更新数不符")
        norm = sum(float(((a - b) ** 2).sum()) for a, b in zip(model.parameters(), original.parameters(), strict=True)) ** .5
        changes.append(ParameterChange(ant=ant, final_updates=update, change_norm=norm))
    return changes


def run(config: Config) -> Result:
    count = verify_manifest(config.input_directory, config.manifest)
    root = config.input_directory
    plan = CoursePlan.model_validate_json((root / "config.json").read_text())
    if plan != CoursePlan.model_validate_json(config.protocol.read_text()):
        raise ValueError("实际与冻结协议不同")
    execution = Execution.model_validate_json((root / "execution.json").read_text())
    if execution.smoke or not execution.completed_at:
        raise ValueError("短流程或未完成执行")
    sources = [SourceRecord.model_validate_json(line) for line in (root / "sources.jsonl").read_text().splitlines()]
    if len(sources) != 1 + 2 * plan.environment.ants or len({s.path for s in sources}) != len(sources):
        raise ValueError("来源数量错误")
    for source in sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError("源模型文件改变")
    training = [WorldRecord.model_validate_json(line) for line in (root / "training/worlds.jsonl").read_text().splitlines()]
    if len(training) != len(plan.training_seeds):
        raise ValueError("训练回合不完整")
    frames = 0
    previous = [0] * plan.environment.ants
    for episode, (row, seed) in enumerate(zip(training, plan.training_seeds, strict=True), 1):
        if (row.model, row.seed, row.task) != ("training", seed, "food" if episode % plan.food_every == 0 else "empty"):
            raise ValueError("训练课程顺序错误")
        if any(a < b for a, b in zip(row.updates, previous, strict=True)):
            raise ValueError("累计训练更新数回退")
        previous = row.updates
        frames += audit_world(root / "training", row, plan.environment, frozen=False)
    changes = snapshots(root, plan, execution, training)
    worlds = [WorldRecord.model_validate_json(line) for line in (root / "evaluation/worlds.jsonl").read_text().splitlines()]
    models = ("recurrent-before", "recurrent-after", "mlp", "rules")
    identities = {(seed, task, name) for seed in plan.evaluation_seeds for task in ("food", "empty") for name in models}
    if len(worlds) != len(identities) or {(r.seed, r.task, r.model) for r in worlds} != identities:
        raise ValueError("评价身份重复或缺失")
    frames += sum(audit_world(root / "evaluation", row, plan.environment) for row in worlds)
    groups = []
    for name in models:
        food = [r for r in worlds if r.model == name and r.task == "food"]
        empty = [r for r in worlds if r.model == name and r.task == "empty"]
        complete = [float(r.steps) for r in food if r.deliveries == plan.environment.stock]
        groups.append(Group(model=name, delivery_fraction=describe([r.deliveries / plan.environment.stock for r in food]),
            complete_food_worlds=len(complete), completion_steps=describe(complete) if complete else None,
            returning_fraction=describe([r.returning_individuals / plan.environment.ants for r in empty]),
            exhausted_fraction=describe([r.exhausted / plan.environment.ants for r in empty]),
            exploration_radius=describe([r.mean_max_radius for r in empty])))
    by_id = {(r.seed, r.task, r.model): r for r in worlds}
    food_pairs = [(by_id[s, "food", "recurrent-after"], by_id[s, "food", "recurrent-before"]) for s in plan.evaluation_seeds]
    empty_pairs = [(by_id[s, "empty", "recurrent-after"], by_id[s, "empty", "recurrent-before"]) for s in plan.evaluation_seeds]
    complete = [float(a.steps - b.steps) for a, b in food_pairs if a.deliveries == b.deliveries == plan.environment.stock]
    contrast = Contrast(delivery_fraction=describe([(a.deliveries - b.deliveries) / plan.environment.stock for a, b in food_pairs]),
        returning_fraction=describe([(a.returning_individuals - b.returning_individuals) / plan.environment.ants for a, b in empty_pairs]),
        exhausted_fraction=describe([(a.exhausted - b.exhausted) / plan.environment.ants for a, b in empty_pairs]),
        exploration_radius=describe([a.mean_max_radius - b.mean_max_radius for a, b in empty_pairs]),
        completion_steps=describe(complete) if complete else None, complete_pairs=len(complete))
    before, after = groups[:2]
    food_ready = after.delivery_fraction.mean >= .9
    return_ready = after.returning_fraction.mean >= max(.5, before.returning_fraction.mean - .05)
    exploration_ready = after.exploration_radius.mean >= 4.5
    result = Result(files_verified=count, frames_verified=frames,
        snapshots_verified=len(execution.snapshot_episodes) * plan.environment.ants, execution=execution,
        worlds=worlds, groups=groups, after_minus_before=contrast, parameter_changes=changes,
        retain_food=food_ready, retain_return=return_ready, exploration_ready=exploration_ready,
        adopt_final=food_ready and return_ready and exploration_ready)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(Config.model_validate_json(args.config.read_text()))
    print(result.model_dump_json(exclude={"worlds"}, indent=2))
