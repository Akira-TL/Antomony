"""核验全部课程原始轨迹后按结果前门槛生成小型汇总。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from mathhackson.training.comparison.auditing import verify_manifest
from mathhackson.training.foraging.plastic_course.assessment import assess
from mathhackson.training.foraging.plastic_course.auditing import (
    FileIdentity, Summary, audit_snapshots, audited_rows, freeze_inputs,
)
from mathhackson.training.foraging.plastic_course.run import Completion, Protocol, verified


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
