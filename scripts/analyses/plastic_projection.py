"""两种方向读出及训练前后的配对汇总；只使用已固定原件。"""
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
from mathhackson.training.foraging.plastic_course.run import Completion, EvaluationRow, Protocol


class EvaluationSet(BaseModel):
    direction_mode: str
    structure_update: int
    assessments: list[SeedAssessment]
    rows: list[EvaluationRow]


class AcrossContrast(BaseModel):
    initialization: int
    condition: Condition
    comparison: Literal["architecture", "training"]
    differences: list[float]
    last_quarter_differences: list[float]
    mean_difference: float
    last_quarter_difference: float


class Decision(BaseModel):
    initialization: int
    original_conditions: bool
    architecture_gain: bool
    training_gain: bool
    passed: bool


class SnapshotSet(BaseModel):
    direction_mode: str
    records: list[SnapshotAudit]


class Summary(BaseModel):
    source: str
    manifest_sha256: str
    source_files: list[FileIdentity]
    snapshots: list[SnapshotSet]
    evaluations: list[EvaluationSet]
    evaluation_frames: int
    contrasts: list[AcrossContrast]
    decisions: list[Decision]
    all_passed: bool
    boundary: str = "两个初始化的合成方向课程；不等于真实搬运、生存或未见复杂环境优势"


def contrast(first: list[EvaluationRow], second: list[EvaluationRow], *, initialization: int,
             condition: Condition, comparison: Literal["architecture", "training"]) -> AcrossContrast:
    def selected(rows: list[EvaluationRow]) -> list[EvaluationRow]:
        return sorted((row for row in rows if row.initialization == initialization
            and row.condition == condition and row.mode == "learned"), key=lambda row: row.batch_seed)
    left, right = selected(first), selected(second)
    if not left or [row.batch_seed for row in left] != [row.batch_seed for row in right]:
        raise ValueError("跨结构或训练时点的配对身份不匹配")
    differences, last = [], []
    for a, b in zip(left, right, strict=True):
        if len(a.mean_after_feedback) != len(b.mean_after_feedback):
            raise ValueError("跨结构或训练时点的回合数不匹配")
        differences.extend((np.asarray(a.mean_after_feedback) - b.mean_after_feedback).tolist())
        last.extend((np.asarray(a.mean_last_quarter) - b.mean_last_quarter).tolist())
    return AcrossContrast(initialization=initialization, condition=condition, comparison=comparison,
        differences=differences, last_quarter_differences=last,
        mean_difference=float(np.mean(differences)), last_quarter_difference=float(np.mean(last)))


def decide(initialization: int, original: SeedAssessment, contrasts: list[AcrossContrast]) -> Decision:
    def qualifies(comparison: str) -> bool:
        rows = {row.condition: row for row in contrasts
            if row.initialization == initialization and row.comparison == comparison}
        if set(rows) != set(CONDITIONS):
            raise ValueError("判定缺少完整条件")
        return (rows["association"].mean_difference >= .03
            and rows["association"].last_quarter_difference >= .03
            and rows["reference"].mean_difference >= -.02
            and rows["reference"].last_quarter_difference >= -.02)
    architecture, training = qualifies("architecture"), qualifies("training")
    return Decision(initialization=initialization, original_conditions=original.passed,
        architecture_gain=architecture, training_gain=training,
        passed=original.passed and architecture and training)


def audit_pairing(root: Path, plan: Protocol) -> None:
    for initialization in plan.initializations:
        seed_path = f"seed-{initialization.seed}"
        before = [load_checkpoint(root / mode / seed_path / "step-0000.npz") for mode in ("unit", "bounded")]
        if not all(torch.equal(value, before[1].state_dict()[key]) for key, value in before[0].state_dict().items()):
            raise ValueError("两种架构并非相同初始参数")
        for mode, trained in zip(("unit", "bounded"), before, strict=True):
            initial_path = root / mode / "untrained" / seed_path / "step-0000.npz"
            initial = load_checkpoint(initial_path)
            with np.load(initial_path, allow_pickle=False) as stored:
                if int(stored["update"]) != 0 or int(stored["seed"]) != initialization.seed:
                    raise ValueError("训练前参照的结构身份不匹配")
            if initial.direction_mode != mode or not all(
                    torch.equal(value, initial.state_dict()[key]) for key, value in trained.state_dict().items()):
                raise ValueError("训练前评价模型并非同一初始结构")
        for condition in CONDITIONS:
            for batch in range(plan.evaluation_batches):
                name = f"{condition}-{initialization.evaluation_seed + batch}-off.npz"
                paths = [root / mode / stage / seed_path / name
                    for mode in ("unit", "bounded") for stage in (Path(), Path("untrained"))]
                with np.load(paths[0], allow_pickle=False) as reference:
                    for path in paths[1:]:
                        with np.load(path, allow_pickle=False) as other:
                            for key in ("observations", "base", "targets", "cue_ids", "reward_flips",
                                        "probabilities", "directions", "rewards"):
                                if not np.array_equal(reference[key], other[key]):
                                    raise ValueError("关闭写入的外源课程或行动跨架构/时点不一致")


def run(root: Path, unit_path: Path, bounded_path: Path, manifest: Path) -> Summary:
    verify_manifest(root, manifest)
    plans = {mode: Protocol.model_validate_json(path.read_bytes())
        for mode, path in (("unit", unit_path), ("bounded", bounded_path))}
    if (plans["unit"].model_dump(exclude={"direction_mode"}) != plans["bounded"].model_dump(exclude={"direction_mode"})
            or not all(plan.direction_mode == mode and plan.evaluate_initial for mode, plan in plans.items())):
        raise ValueError("协议除方向模式外不一致，或缺少初始参照")
    evaluations, snapshots = [], []
    for mode, path in (("unit", unit_path), ("bounded", bounded_path)):
        plan, directory = plans[mode], root / mode
        completion = Completion.model_validate_json((directory / "completion.json").read_bytes())
        if not completion.complete or completion.initializations_completed != 2:
            raise ValueError("当前架构实施不完整，不能判断通过")
        if (directory / "protocol.json").read_bytes() != path.read_bytes():
            raise ValueError("实际协议偏离冻结版本")
        snapshots.append(SnapshotSet(direction_mode=mode, records=audit_snapshots(directory, plan)))
        for update, source in ((0, directory / "untrained"), (plan.updates, directory)):
            rows = audited_rows(source, plan, structure_update=update)
            evaluations.append(EvaluationSet(direction_mode=mode, structure_update=update, rows=rows,
                assessments=[assess(rows, initialization=item.seed, steps=plan.course.steps) for item in plan.initializations]))
    plan = plans["bounded"]
    audit_pairing(root, plan)
    by_key = {(item.direction_mode, item.structure_update): item for item in evaluations}
    final = by_key[("bounded", plan.updates)]
    contrasts = [contrast(final.rows, by_key[key].rows, initialization=item.seed,
        condition=condition, comparison=comparison)
        for item in plan.initializations for condition in CONDITIONS
        for comparison, key in (("architecture", ("unit", plan.updates)), ("training", ("bounded", 0)))]
    decisions = [decide(item.initialization, item, contrasts) for item in final.assessments]
    files = [FileIdentity(path=str(path.relative_to(root)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob("*")) if path.is_file()]
    return Summary(source=str(root), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_files=files, snapshots=snapshots, evaluations=evaluations,
        evaluation_frames=sum(len(item.rows) for item in evaluations) * plan.course.steps * plan.course.batch,
        contrasts=contrasts, decisions=decisions, all_passed=all(item.passed for item in decisions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--unit-plan", type=Path)
    parser.add_argument("--bounded-plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-inputs", action="store_true")
    args = parser.parse_args()
    if args.freeze_inputs:
        if args.manifest or args.unit_plan or args.bounded_plan:
            parser.error("清单固定与效果汇总必须分开")
        freeze_inputs(args.input, args.output)
        print("已固定完整输入清单；尚未汇总效果", flush=True)
        return
    if args.manifest is None or args.unit_plan is None or args.bounded_plan is None:
        parser.error("汇总必须提供固定清单及两个协议")
    result = run(args.input, args.unit_plan, args.bounded_plan, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(exclude={"source_files", "evaluations"}), flush=True)


if __name__ == "__main__":
    main()
