"""近似规模基础资格的完整性核对与配对描述。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.direction.checkpoint import load_motor
from .auditing import Estimate, audit_world, describe, verify_manifest
from .matched_foundation import Execution, MatchedPlan
from .run import SignalRecord, SourceRecord, WorldRecord

MODELS = ("mlp-matched", "mlp-original", "rules")


class Group(BaseModel):
    model: str
    delivery_fraction: Estimate
    complete_food_worlds: int
    completion_steps: Estimate | None
    empty_returning_fraction: Estimate
    empty_exhausted_fraction: Estimate
    empty_max_radius: Estimate
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
    snapshots_verified: int
    optimizers_verified: int
    execution: Execution
    worlds: list[WorldRecord]
    groups: list[Group]
    matched_minus_comparator: list[Contrast]
    all_minimum_ready: bool
    equivalence_established: bool = False


def scores(worlds: list[WorldRecord], plan: MatchedPlan) -> tuple[list[Group], list[Contrast]]:
    identities = {(seed, task, name) for seed in plan.evaluation_seeds for task in ("food", "empty") for name in MODELS}
    by_id = {(r.seed, r.task, r.model): r for r in worlds}
    if set(by_id) != identities or len(worlds) != len(identities):
        raise ValueError("世界身份重复、缺失或多余")
    ants, stock = plan.environment.ants, plan.environment.stock
    if stock < 1:
        raise ValueError("需要非零食物任务")
    if any(not 0 <= r.deliveries <= r.pickups <= stock or not 0 <= r.returning_individuals <= ants
           or not 0 <= r.exhausted <= ants or r.budget_returns < r.returning_individuals for r in worlds):
        raise ValueError("事件计数越界")
    groups = []
    for name in MODELS:
        food = [by_id[seed, "food", name] for seed in plan.evaluation_seeds]
        empty = [by_id[seed, "empty", name] for seed in plan.evaluation_seeds]
        complete = [float(r.steps) for r in food if r.deliveries == stock]
        delivery = describe([r.deliveries / stock for r in food])
        returning = describe([r.returning_individuals / ants for r in empty])
        radius = describe([r.mean_max_radius for r in empty])
        groups.append(Group(model=name, delivery_fraction=delivery, complete_food_worlds=len(complete),
            completion_steps=describe(complete) if complete else None, empty_returning_fraction=returning,
            empty_exhausted_fraction=describe([r.exhausted / ants for r in empty]), empty_max_radius=radius,
            minimum_ready=(delivery.mean >= plan.minimum_delivery_fraction
                           and returning.mean >= plan.minimum_returning_fraction
                           and radius.mean >= plan.minimum_exploration_radius)))
    contrasts = []
    for name in MODELS[1:]:
        food = [(by_id[seed, "food", MODELS[0]], by_id[seed, "food", name]) for seed in plan.evaluation_seeds]
        empty = [(by_id[seed, "empty", MODELS[0]], by_id[seed, "empty", name]) for seed in plan.evaluation_seeds]
        complete = [float(a.steps - b.steps) for a, b in food if a.deliveries == b.deliveries == stock]
        contrasts.append(Contrast(comparator=name,
            delivery_fraction=describe([(a.deliveries - b.deliveries) / stock for a, b in food]),
            paired_completion_steps=describe(complete) if complete else None, complete_pairs=len(complete),
            empty_returning_fraction=describe([(a.returning_individuals - b.returning_individuals) / ants for a, b in empty]),
            empty_exhausted_fraction=describe([(a.exhausted - b.exhausted) / ants for a, b in empty]),
            empty_max_radius=describe([a.mean_max_radius - b.mean_max_radius for a, b in empty])))
    return groups, contrasts


def audit_training(directory: Path, plan: MatchedPlan) -> tuple[int, int]:
    steps = sorted({0, *range(plan.checkpoint_every, plan.signal_updates + 1, plan.checkpoint_every), plan.signal_updates})
    expected_paths = {directory / f"seed-{seed}-signal-{step:06d}.npz" for seed in plan.model_seeds for step in steps}
    if set(directory.glob("*.npz")) != expected_paths:
        raise ValueError("参数快照缺失或多余")
    expected_optimizers = {path.with_suffix(".optimizer.pt") for path in expected_paths if not path.name.endswith("-000000.npz")}
    if set(directory.glob("*.optimizer.pt")) != expected_optimizers:
        raise ValueError("优化器快照缺失或多余")
    for seed in plan.model_seeds:
        initial = FeedforwardPolicy(seed, hidden_width=plan.hidden_width)
        for step in steps:
            path = directory / f"seed-{seed}-signal-{step:06d}.npz"
            with np.load(path, allow_pickle=False) as saved:
                if int(saved["update"]) != step or str(saved["phase"]) != "signal":
                    raise ValueError("参数阶段或轮次不符")
            model = FeedforwardPolicy.load(path)
            if model.hidden_width != plan.hidden_width:
                raise ValueError("模型宽度不符")
            model.assert_reserved()
            for name, value in model.state_dict().items():
                if (step == 0 or name.startswith("value.")) and not value.equal(initial.state_dict()[name]):
                    raise ValueError("初始化或冻结价值参数改变")
            if step:
                optimizer = torch.load(path.with_suffix(".optimizer.pt"), weights_only=True, map_location="cpu")
                groups, states = optimizer["param_groups"], optimizer["state"]
                if len(groups) != 1 or len(states) != 6 or groups[0]["params"] != list(range(6)):
                    raise ValueError("优化器参数身份错误")
                if groups[0]["lr"] != plan.signal_rate or groups[0]["weight_decay"] != plan.weight_decay:
                    raise ValueError("优化器超参数改变")
                for index, parameter in enumerate(list(model.parameters())[:6]):
                    state = states[index]
                    if int(state["step"]) != step:
                        raise ValueError("优化器步数不符")
                    for name in ("exp_avg", "exp_avg_sq"):
                        if state[name].shape != parameter.shape or not bool(torch.isfinite(state[name]).all()):
                            raise ValueError("优化器状态维度或数值非法")
    records = [SignalRecord.model_validate_json(line) for line in (directory / "signal.jsonl").read_text().splitlines()]
    expected = {(seed, step) for seed in plan.model_seeds for step in steps if step}
    if len(records) != len(expected) or {(r.seed, r.update) for r in records} != expected:
        raise ValueError("训练记录缺失或重复")
    if any(not np.isfinite([r.loss, r.mean_angle, r.within_15]).all() or not 0 <= r.within_15 <= 1
           or not 0 <= r.mean_angle <= 180 for r in records):
        raise ValueError("训练诊断非法")
    return len(expected_paths), len(expected_optimizers)


def audit(directory: Path, manifest: Path, protocol: Path) -> Result:
    files = verify_manifest(directory, manifest)
    plan = MatchedPlan.model_validate_json((directory / "config.json").read_text())
    if plan != MatchedPlan.model_validate_json(protocol.read_text()):
        raise ValueError("实际配置与冻结协议不符")
    execution = Execution.model_validate_json((directory / "execution.json").read_text())
    if execution.smoke or not execution.completed_at or execution.evaluation_updates or execution.training_updates_per_individual != plan.signal_updates:
        raise ValueError("执行未完成、含短流程或更新次数错误")
    sources = [SourceRecord.model_validate_json(line) for line in (directory / "sources.jsonl").read_text().splitlines()]
    expected = {plan.motor, *[str(Path(plan.original_directory) / f"seed-{seed}-signal-002400.npz") for seed in plan.model_seeds]}
    if len(sources) != len(expected) or {s.path for s in sources} != expected:
        raise ValueError("来源身份缺失或多余")
    if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in sources):
        raise ValueError("源模型改变")
    motor, _ = load_motor(Path(plan.motor))
    if len(execution.models) != 2 or [m.model for m in execution.models] != list(MODELS[:2]):
        raise ValueError("模型分类记录错误")
    for recorded, width in zip(execution.models, (plan.hidden_width, 14), strict=True):
        if (recorded.policy_parameters != 95 * width + 35
                or recorded.motor_parameters != sum(p.numel() for p in motor.parameters())
                or recorded.reserved_connections != 50 * width or recorded.frozen_value_parameters != 9):
            raise ValueError("模型参数分类不符")
    snapshots, optimizers = audit_training(directory, plan)
    worlds = [WorldRecord.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
    groups, contrasts = scores(worlds, plan)
    frames = sum(audit_world(directory, row, plan.environment) for row in worlds)
    if files != snapshots + optimizers + len(worlds) + 5:
        raise ValueError("原始目录文件数量异常")
    return Result(files_verified=files, frames_verified=frames, snapshots_verified=snapshots,
        optimizers_verified=optimizers, execution=execution, worlds=worlds, groups=groups,
        matched_minus_comparator=contrasts, all_minimum_ready=all(g.minimum_ready for g in groups))
