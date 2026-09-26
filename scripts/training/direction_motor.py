"""训练纯方向动作基础并保存每个指定间隔的权重及完整验收轨迹。"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import torch
from pydantic import BaseModel

from mathhackson.training.direction.checkpoint import MotorSnapshot, load_motor, save_motor
from mathhackson.training.direction.curriculum import MotorConfig, evaluate_motor, train_motor
from mathhackson.training.direction.policy import DirectionMotor


class SeedResult(BaseModel):
    seed: int
    checkpoint: str
    sha256: str
    parameters: int
    passed: bool
    trials_passed: int
    trials: int
    max_steady_error_degrees: float
    min_steady_progress_fraction: float


class RunConfig(BaseModel):
    training: MotorConfig
    seeds: list[int]
    evaluation_seed: int = 8001
    source_commit: str
    torch_version: str
    threads: int
    status: str = "开发阶段动作接口验收，不是自训练效果实验"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("训练种子不可重复")
    config = MotorConfig(steps=args.steps, checkpoint_every=args.checkpoint_every)
    torch.set_num_threads(1)
    directory: Path = args.output
    directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run = RunConfig(training=config, seeds=args.seeds, source_commit=commit,
                    torch_version=torch.__version__, threads=torch.get_num_threads())
    (directory / "config.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    results: list[SeedResult] = []
    for seed in args.seeds:
        output = directory / f"seed-{seed}"
        output.mkdir()

        def save(model: DirectionMotor, snapshot: MotorSnapshot) -> None:
            save_motor(output / f"update-{snapshot.updates:06d}.npz", model, snapshot)
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(snapshot.model_dump_json() + "\n")

        train_motor(seed, config, save)
        checkpoint = output / f"update-{config.steps:06d}.npz"
        model, _ = load_motor(checkpoint)
        evaluation = evaluate_motor(model, run.evaluation_seed)
        (output / "trajectories.json").write_text(evaluation.model_dump_json(), encoding="utf-8")
        result = SeedResult(seed=seed, checkpoint=str(checkpoint),
                            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            parameters=sum(p.numel() for p in model.parameters()),
                            passed=evaluation.passed,
                            trials_passed=sum(t.passed for t in evaluation.trials),
                            trials=len(evaluation.trials),
                            max_steady_error_degrees=evaluation.max_steady_error_degrees,
                            min_steady_progress_fraction=evaluation.min_steady_progress_fraction)
        results.append(result)
        with (directory / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(result.model_dump_json() + "\n")
        print(result.model_dump_json(), flush=True)
    if not all(result.passed for result in results):
        raise SystemExit("存在未通过的动作基础，原始结果已保留，不自动挑选成功种子")


if __name__ == "__main__":
    main()
