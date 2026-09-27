"""同情境独立试行基线的完整身份、配对及固定判据审计。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.comparison.auditing import verify_manifest
from mathhackson.training.foraging.feedback_cues import CONDITIONS, Condition
from mathhackson.training.foraging.plastic_course.assessment import SeedAssessment, assess
from mathhackson.training.foraging.plastic_course.auditing import (
    FileIdentity, SnapshotAudit, audit_snapshots, audited_rows, freeze_inputs,
)
from mathhackson.training.foraging.plastic_course.checkpoint import load_checkpoint
from mathhackson.training.foraging.plastic_course.run import Completion, EvaluationRow, Protocol, TrainingRow, verified


class ArmResult(BaseModel):
    mode: Literal["history", "paired"]
    snapshots: list[SnapshotAudit]
    initial: list[EvaluationRow]
    trained: list[EvaluationRow]
    initial_assessments: list[SeedAssessment]
    trained_assessments: list[SeedAssessment]


class Difference(BaseModel):
    initialization: int
    condition: Condition
    comparison: Literal["baseline", "training"]
    after: list[float]
    last: list[float]
    mean_after: float
    mean_last: float


class Decision(BaseModel):
    initialization: int
    original_conditions: bool
    baseline_gain: bool
    training_gain: bool
    passed: bool


class Summary(BaseModel):
    source: str
    manifest_sha256: str
    source_files: list[FileIdentity]
    arms: list[ArmResult]
    evaluation_frames: int
    contrasts: list[Difference]
    decisions: list[Decision]
    all_passed: bool
    boundary: str = "两个初始化的合成方向课程，不证明实际蚁群、生存或广泛未知环境优势"


def compare(first: list[EvaluationRow], second: list[EvaluationRow], *, seed: int,
            condition: Condition, comparison: Literal["baseline", "training"]) -> Difference:
    def choose(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        return sorted((r for r in rows if r.initialization == seed and r.condition == condition
            and r.mode == "learned"), key=lambda r: r.batch_seed)
    left, right = choose(first), choose(second)
    if not left or [r.batch_seed for r in left] != [r.batch_seed for r in right]:
        raise ValueError("跨基线或时点的配对身份不一致")
    after, last = [], []
    for a, b in zip(left, right, strict=True):
        for name in ("mean_after_feedback", "mean_last_quarter"):
            if len(getattr(a, name)) != len(getattr(b, name)):
                raise ValueError("配对回合数量不一致")
        after.extend((np.asarray(a.mean_after_feedback) - b.mean_after_feedback).tolist())
        last.extend((np.asarray(a.mean_last_quarter) - b.mean_last_quarter).tolist())
    return Difference(initialization=seed, condition=condition, comparison=comparison,
        after=after, last=last, mean_after=float(np.mean(after)), mean_last=float(np.mean(last)))


def decide(original: SeedAssessment, contrasts: list[Difference]) -> Decision:
    def qualifies(comparison: str) -> bool:
        rows = [r for r in contrasts if r.initialization == original.initialization and r.comparison == comparison]
        by_condition = {r.condition: r for r in rows}
        if len(rows) != 3 or set(by_condition) != set(CONDITIONS):
            raise ValueError("固定判据缺少完整且唯一的条件")
        associated, reference = by_condition["association"], by_condition["reference"]
        return (associated.mean_after >= .03 and associated.mean_last >= .03
            and reference.mean_after >= -.02 and reference.mean_last >= -.02)
    baseline, training = qualifies("baseline"), qualifies("training")
    return Decision(initialization=original.initialization, original_conditions=original.passed,
        baseline_gain=baseline, training_gain=training, passed=original.passed and baseline and training)


def audit_pairing(root: Path, plan: Protocol) -> None:
    for init in plan.initializations:
        folder = f"seed-{init.seed}"
        initial = []
        for mode in ("history", "paired"):
            for stage in (Path(), Path("untrained")):
                path = root / mode / stage / folder / "step-0000.npz"
                model = load_checkpoint(path)
                with np.load(path, allow_pickle=False) as saved:
                    if int(saved["seed"]) != init.seed or int(saved["update"]) != 0:
                        raise ValueError("初始结构身份不匹配")
                if model.direction_mode != "bounded":
                    raise ValueError("基线比较不允许改变方向模式")
                initial.append(model)
        for other in initial[1:]:
            if not all(torch.equal(value, other.state_dict()[name]) for name, value in initial[0].state_dict().items()):
                raise ValueError("两基线及初始参照的参数不同")
        for condition in CONDITIONS:
            for batch in range(plan.evaluation_batches):
                name = f"{condition}-{init.evaluation_seed + batch}-off.npz"
                paths = [root / mode / stage / folder / name
                    for mode in ("history", "paired") for stage in (Path(), Path("untrained"))]
                with np.load(paths[0], allow_pickle=False) as first:
                    for path in paths[1:]:
                        with np.load(path, allow_pickle=False) as other:
                            for key in ("observations", "base", "targets", "cue_ids", "reward_flips",
                                        "probabilities", "directions", "rewards"):
                                if not np.array_equal(first[key], other[key]):
                                    raise ValueError("关闭写入的课程或行动跨基线/时点改变")


def audit_training_pairs(directory: Path, plan: Protocol) -> None:
    updates = sorted({plan.updates, *range(plan.checkpoint_interval, plan.updates + 1, plan.checkpoint_interval)})
    for init in plan.initializations:
        source = directory / f"seed-{init.seed}"
        rows = [TrainingRow.model_validate_json(line) for line in (source / "training.jsonl").read_text().splitlines()]
        if [row.update for row in rows] != list(range(1, plan.updates + 1)):
            raise ValueError("训练轮次不完整或重复")
        if any(not np.isfinite([r.loss, r.mean_reward, r.accepted_fraction, r.gradient_norm, r.elapsed_seconds]).all()
                or not 0 <= r.accepted_fraction <= 1 or r.gradient_norm < 0 or r.elapsed_seconds < 0 for r in rows):
            raise ValueError("训练记录包含无效数值")
        if set(source.glob("training-*.npz")) != {source / f"training-{u:04d}.npz" for u in updates}:
            raise ValueError("训练片段保存点不完整")
        for update in updates:
            with np.load(source / f"training-{update:04d}.npz", allow_pickle=False) as trace:
                if int(trace["structure_update"]) != update - 1 or trace["rewards"].shape != (plan.course.steps, plan.course.batch):
                    raise ValueError("训练片段结构或形状不符")
                for name in ("observations", "targets", "cue_ids", "reward_flips"):
                    if not np.array_equal(trace[name][:, 0::2], trace[name][:, 1::2]):
                        raise ValueError("训练配对没有使用同一情境")
                if not np.allclose(trace["base"][:, 0::2], trace["base"][:, 1::2], atol=1e-6, rtol=0):
                    raise ValueError("配对情境的冻结方向超出float32舍入容差")
                if not np.array_equal(trace["conditions"][0::2], trace["conditions"][1::2]):
                    raise ValueError("训练配对条件不一致")


def run(root: Path, history: Path, paired: Path, manifest: Path) -> Summary:
    verify_manifest(root, manifest)
    plans = {mode: Protocol.model_validate_json(path.read_bytes())
        for mode, path in (("history", history), ("paired", paired))}
    if (plans["history"].model_dump(exclude={"baseline_mode"}) != plans["paired"].model_dump(exclude={"baseline_mode"})
            or not all(p.baseline_mode == mode and p.paired_courses and p.evaluate_initial
                and p.direction_mode == "bounded" for mode, p in plans.items())):
        raise ValueError("两协议必须仅改变基线方式并保留配对课程和初始参照")
    arms = []
    for mode, path in (("history", history), ("paired", paired)):
        directory, plan = root / mode, plans[mode]
        completion = Completion.model_validate_json((directory / "completion.json").read_bytes())
        if not completion.complete or completion.initializations_completed != 2:
            raise ValueError("实施不完整，不作通过判定")
        if (directory / "protocol.json").read_bytes() != path.read_bytes():
            raise ValueError("原始协议不同于冻结协议")
        for init in plan.initializations:
            verified(init.foundation)
        audit_training_pairs(directory, plan)
        snapshots = audit_snapshots(directory, plan)
        initial = audited_rows(directory / "untrained", plan, structure_update=0)
        trained = audited_rows(directory, plan)
        arms.append(ArmResult(mode=mode, snapshots=snapshots, initial=initial, trained=trained,
            initial_assessments=[assess(initial, initialization=i.seed, steps=plan.course.steps) for i in plan.initializations],
            trained_assessments=[assess(trained, initialization=i.seed, steps=plan.course.steps) for i in plan.initializations]))
    plan = plans["paired"]
    audit_pairing(root, plan)
    control, candidate = arms
    contrasts = [compare(candidate.trained, second, seed=init.seed, condition=condition, comparison=comparison)
        for init in plan.initializations for condition in CONDITIONS
        for comparison, second in (("baseline", control.trained), ("training", candidate.initial))]
    decisions = [decide(result, contrasts) for result in candidate.trained_assessments]
    files = [FileIdentity(path=str(path.relative_to(root)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob("*")) if path.is_file()]
    return Summary(source=str(root), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_files=files, arms=arms, evaluation_frames=sum(len(a.initial) + len(a.trained) for a in arms)
        * plan.course.steps * plan.course.batch, contrasts=contrasts, decisions=decisions,
        all_passed=all(d.passed for d in decisions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--history-plan", type=Path)
    parser.add_argument("--paired-plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-inputs", action="store_true")
    args = parser.parse_args()
    if args.freeze_inputs:
        if args.manifest or args.history_plan or args.paired_plan:
            parser.error("清单固定必须与结果汇总分开")
        freeze_inputs(args.input, args.output)
        print("已固定输入，尚未汇总效果", flush=True)
        return
    if args.manifest is None or args.history_plan is None or args.paired_plan is None:
        parser.error("必须指定清单与两份协议")
    result = run(args.input, args.history_plan, args.paired_plan, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(include={"evaluation_frames", "decisions", "all_passed", "boundary"}), flush=True)


if __name__ == "__main__":
    main()
