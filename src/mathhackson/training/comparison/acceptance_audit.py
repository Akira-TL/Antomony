"""固定课程分析前的身份、帧与冻结参数检查，不计算留出选择收益。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.foraging.candidate_probe import ParentFrame
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig
from mathhackson.training.foraging.update_curriculum import CurriculumExecution, CurriculumPlan, CurriculumWorld, LabeledDecision, world_key
from .auditing import verify_manifest


class Audit(BaseModel):
    files: int
    snapshots: int
    frames: int


def audit_inputs(directory: Path, manifest: Path, protocol: Path) -> tuple[CurriculumExecution, Audit]:
    count = verify_manifest(directory, manifest)
    execution = CurriculumExecution.model_validate_json((directory / "execution.json").read_text())
    plan = execution.plan
    if (execution.smoke or plan != CurriculumPlan.model_validate_json(protocol.read_text())
            or execution.plan_sha256 != hashlib.sha256(protocol.read_bytes()).hexdigest()
            or plan.probe.policy_kind != "mlp-memory" or plan.probe.adaptation.feedback_mode != "observed-window"):
        raise ValueError("执行身份与固定的新模型协议不符")
    sources = {plan.probe.policy_path(i) for i in range(plan.probe.environment.ants)} | {Path(plan.probe.motor_path)}
    if len(execution.sources) != len(sources) or {Path(item.path) for item in execution.sources} != sources:
        raise ValueError("来源模型缺失或重复")
    for item in execution.sources:
        if hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256:
            raise ValueError("源参数散列改变")
    bases = [MemoryPolicy.load(plan.probe.policy_path(i)).state_dict() for i in range(plan.probe.environment.ants)]
    worlds = [CurriculumWorld.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
    expected = {world_key(i, seed) for i in range(len(plan.initial_norms)) for seed in plan.train_seeds + plan.held_seeds}
    if len(worlds) != len(expected) or {w.key for w in worlds} != expected:
        raise ValueError("世界数量或身份错误")
    snapshots = frames = 0
    for world in worlds:
        index = plan.initial_norms.index(world.initial_norm)
        if (world.key != world_key(index, world.result.seed) or world.result.deaths or world.result.condition != "benign"
                or world.partition != ("train" if world.result.seed in plan.train_seeds else "held")
                or not 1 <= world.result.steps <= plan.probe.environment.horizon):
            raise ValueError("世界条件、时限或分区错误")
        folder = directory / world.key
        with (folder / "pairs.jsonl").open() as stream:
            if sum(1 for _ in stream) != world.result.pairs or world.result.pairs > plan.probe.pairs_per_world:
                raise ValueError("候选记录数量不符合预算")
        tick = 0
        with gzip.open(folder / "parent.jsonl.gz", "rt") as stream:
            for tick, line in enumerate(stream, 1):
                frame = ParentFrame.model_validate_json(line)
                if (frame.tick != tick or any(len(v) != plan.probe.environment.ants for v in (frame.moves, frame.turns, frame.rewards))
                        or not np.isfinite(frame.turns + frame.rewards).all()):
                    raise ValueError("父轨迹帧无效")
        if tick != world.result.steps:
            raise ValueError("父轨迹长度不符")
        frames += tick
        for focal, base in enumerate(bases):
            initial = np.zeros(45, dtype=np.float32)
            if world.initial_norm:
                if plan.probe.initial_residual_pattern != "coherent-fourth":
                    raise ValueError("本核验只用于固定的同向第四接收器扰动")
                value = np.zeros((9, 5))
                value[:, 0] = np.random.default_rng(world.result.seed * 8 + focal).choice((-1., 1.))
                initial = (value.flatten() * (world.initial_norm / np.linalg.norm(value))).astype(np.float32)
            ticks = set(range(64, tick + 1, 64)) | {tick}
            if world.initial_norm:
                ticks.add(0)
            paths = {folder / f"tick-{t:04d}-ant-{focal:02d}.residual.npz" for t in ticks}
            if set(folder.glob(f"*-ant-{focal:02d}.residual.npz")) != paths:
                raise ValueError("参数快照缺失或额外")
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    if (str(data["adapter_kind"]) != "direction-trust" or int(data["writes"]) != 0
                            or np.any(data["stable"]) or not np.array_equal(data["fast"], initial)
                            or TrustConfig.model_validate_json(str(data["config_json"])) != plan.probe.adaptation):
                        raise ValueError("父轨迹残差或更新配置改变")
                model_path = path.with_name(path.name.replace(".residual.npz", ".npz"))
                state = MemoryPolicy.load(model_path).state_dict()
                if any(not torch.equal(state[key], value) for key, value in base.items()):
                    raise ValueError("父轨迹基础或记忆参数改变")
                snapshots += 1
    return execution, Audit(files=count, snapshots=snapshots, frames=frames)


def audit_pairs(rows: list[LabeledDecision], plan: CurriculumPlan) -> None:
    for row in rows:
        record = row.record
        if (record.tick % plan.probe.adaptation.window or not 0 < record.tick < plan.probe.environment.horizon
                or record.proposal.steps != plan.probe.adaptation.window or record.diagnostics is None
                or record.diagnostics.mean_kl > plan.probe.adaptation.maximum_mean_kl + 1e-6
                or record.diagnostics.maximum_kl > plan.probe.adaptation.maximum_state_kl + 1e-6
                or not np.isfinite(record.result.accepted_weights).all()
                or len(record.result.accepted_weights) != 45):
            raise ValueError("候选时间、行动约束或参数无效")
        for outcome in (record.result.skip, record.result.accept):
            if (not 0 < outcome.steps <= plan.probe.branch_horizon or outcome.focal_death or outcome.colony_deaths
                    or outcome.focal_injury != 0.):
                raise ValueError("分支超出时限或无危险边界")
