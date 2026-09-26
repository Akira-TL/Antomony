"""核验全部课程原始轨迹后按结果前门槛生成小型汇总。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.comparison.auditing import verify_manifest
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.feedback_cues import CONDITIONS
from mathhackson.training.foraging.plastic_course.assessment import SeedAssessment, assess
from mathhackson.training.foraging.plastic_course.checkpoint import load_checkpoint
from mathhackson.training.foraging.plastic_course.run import Completion, EvaluationRow, Protocol, verified


class FileIdentity(BaseModel):
    path: str
    bytes: int
    sha256: str


class SnapshotAudit(BaseModel):
    initialization: int
    snapshots: int
    parameter_change_norm: float
    memory_change_norm: float


class Summary(BaseModel):
    source: str
    manifest_sha256: str
    source_files: list[FileIdentity]
    snapshots: list[SnapshotAudit]
    evaluation_frames: int
    initialization_results: list[SeedAssessment]
    all_passed: bool
    boundary: str = "两个外层初始化的合成方向课程可行性；不是真实搬运、生存或复杂环境优势"


def freeze_inputs(directory: Path, manifest: Path) -> None:
    if manifest.resolve().is_relative_to(directory.resolve()):
        raise ValueError("输入清单必须位于原始目录之外")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("原始目录为空")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")


def audit_snapshots(directory: Path, plan: Protocol) -> list[SnapshotAudit]:
    motor, _ = load_motor(verified(plan.motor))
    result = []
    for initialization in plan.initializations:
        source = directory / f"seed-{initialization.seed}"
        updates = sorted({0, plan.updates, *range(plan.checkpoint_interval, plan.updates + 1, plan.checkpoint_interval)})
        if set(source.glob("step-*.npz")) != {source / f"step-{update:04d}.npz" for update in updates}:
            raise ValueError("结构保存点缺失或多出未定义版本")
        first = None
        model = None
        for update in updates:
            path = source / f"step-{update:04d}.npz"
            model = load_checkpoint(path)
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["update"]) != update or int(archive["seed"]) != initialization.seed:
                    raise ValueError("结构保存点身份不匹配")
            if model.max_step != plan.max_step or model.max_fast != plan.max_fast or model.feature_width != 45:
                raise ValueError("模型架构约束不同于协议")
            if not all(torch.equal(value, motor.state_dict()[key]) for key, value in model.motor.state_dict().items()):
                raise ValueError("保存点中的动作底座被改变")
            if first is None:
                first = model
        assert first is not None and model is not None
        change = [value - first.state_dict()[key] for key, value in model.state_dict().items() if not key.startswith("motor.")]
        memory = [value - first.state_dict()[key] for key, value in model.state_dict().items()
            if key in ("recent_weights", "sparse_weights")]
        result.append(SnapshotAudit(initialization=initialization.seed, snapshots=len(updates),
            parameter_change_norm=float(torch.cat([value.flatten() for value in change]).norm()),
            memory_change_norm=float(torch.cat([value.flatten() for value in memory]).norm())))
    return result


def audited_rows(directory: Path, plan: Protocol) -> list[EvaluationRow]:
    results = []
    for initialization in plan.initializations:
        source = directory / f"seed-{initialization.seed}"
        expected = {(condition, initialization.evaluation_seed + batch, mode)
            for condition in CONDITIONS for batch in range(plan.evaluation_batches)
            for mode in ("learned", "off", "always", "matched")}
        seen = set()
        rows = [EvaluationRow.model_validate_json(line) for line in (source / "evaluations.jsonl").read_text().splitlines()]
        for row in rows:
            key = (row.condition, row.batch_seed, row.mode)
            if key in seen or key not in expected or row.initialization != initialization.seed:
                raise ValueError("评价身份重复或不符协议")
            seen.add(key)
            prefix = f"{row.condition}-{row.batch_seed}"
            with np.load(source / f"{prefix}-{row.mode}.npz", allow_pickle=False) as trace, np.load(
                    source / f"{prefix}-learned.npz", allow_pickle=False) as learned:
                shape = (plan.course.steps, plan.course.batch)
                if trace["rewards"].shape != shape or int(trace["structure_update"]) != plan.updates:
                    raise ValueError("评价帧数或模型终点不符")
                for name in trace.files:
                    value = trace[name]
                    if value.dtype.kind in "fc" and not np.isfinite(value).all():
                        raise ValueError("轨迹包含非有限值")
                for name in ("observations", "base", "targets", "cue_ids", "reward_flips"):
                    if not np.array_equal(trace[name], learned[name]):
                        raise ValueError("配对组的外源课程不一致")
                if not np.array_equal(trace["probabilities"][:4], learned["probabilities"][:4]):
                    raise ValueError("反馈前基础行为不一致")
                if not np.array_equal(trace["directions"][:4], learned["directions"][:4]):
                    raise ValueError("反馈前动作随机流未配对")
                receptors = trace["observations"][..., :72].reshape(*shape, 9, 8)
                if np.any(receptors[..., 3:]):
                    raise ValueError("基础课程包含保留通道信号")
                if not np.allclose(trace["rewards"], (trace["directions"] * trace["targets"]).sum(axis=-1), atol=1e-6):
                    raise ValueError("反馈并非所选方向的真实得分")
                accepted = trace["accepted"]
                if accepted.dtype != np.bool_ or accepted.shape != shape or accepted[np.arange(shape[0]) % 4 != 3].any():
                    raise ValueError("接受记录或四帧边界无效")
                if row.mode == "off" and (accepted.any() or trace["fast"].any()):
                    raise ValueError("关闭写入对照改变了快权重")
                if row.mode == "always" and not accepted[3::4].all():
                    raise ValueError("始终接受对照缺少判断")
                if row.mode == "matched" and not np.array_equal(accepted.sum(axis=0), learned["accepted"].sum(axis=0)):
                    raise ValueError("随机接受次数没有逐回合匹配")
                fast = trace["fast"]
                change = np.linalg.norm((fast - np.concatenate((np.zeros_like(fast[:1]), fast[:-1]))).reshape(*shape, -1), axis=-1)
                if np.any(change > plan.max_step + 1e-5):
                    raise ValueError("单次写入超过协议上限")
                if (np.any(change[~accepted] != 0) or not np.allclose(change, trace["write_norm"], atol=1e-6)
                        or np.any(np.linalg.norm(fast.reshape(*shape, -1), axis=-1) > plan.max_fast + 1e-5)):
                    raise ValueError("跳过、写入范数或参数约束不符")
                for actual, reported in ((trace["rewards"][4:].mean(axis=0), row.mean_after_feedback),
                        (trace["rewards"][-shape[0] // 4:].mean(axis=0), row.mean_last_quarter),
                        (accepted.sum(axis=0), row.acceptances),
                        ((trace["write_norm"] > 1e-9).sum(axis=0), row.nonzero_writes)):
                    if not np.allclose(actual, reported, atol=1e-6):
                        raise ValueError("汇总与原始记录不一致")
        if seen != expected:
            raise ValueError("评价组合缺失")
        results.extend(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--freeze-inputs", action="store_true")
    args = parser.parse_args()
    if args.freeze_inputs:
        if args.manifest:
            parser.error("建立清单与核验清单须分开执行")
        freeze_inputs(args.input, args.output)
        print(f"已固定文件清单：{args.output}，尚未计算效果结果", flush=True)
        return
    if args.manifest is None:
        parser.error("汇总前必须提供已独立固定的输入清单")
    verify_manifest(args.input, args.manifest)
    completion = Completion.model_validate_json((args.input / "completion.json").read_text())
    if not completion.complete or completion.initializations_completed != 2:
        raise ValueError("训练或评价未完整完成；保留失败，不给完整效果判断")
    raw_plan = (args.input / "protocol.json").read_bytes()
    if raw_plan != args.plan.read_bytes():
        raise ValueError("实施与固定协议不符")
    plan = Protocol.model_validate_json(raw_plan)
    verified(plan.motor)
    for initialization in plan.initializations:
        verified(initialization.foundation)
    snapshots = audit_snapshots(args.input, plan)
    rows = audited_rows(args.input, plan)
    assessments = [assess(rows, initialization=item.seed, steps=plan.course.steps) for item in plan.initializations]
    files = [FileIdentity(path=str(path.relative_to(args.input)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(args.input.rglob("*")) if path.is_file()]
    summary = Summary(source=str(args.input), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_files=files, snapshots=snapshots,
        evaluation_frames=len(rows) * plan.course.batch * plan.course.steps,
        initialization_results=assessments, all_passed=all(item.passed for item in assessments))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(summary.model_dump_json(indent=2))
    print(summary.model_dump_json(exclude={"source_files"}), flush=True)


if __name__ == "__main__":
    main()
