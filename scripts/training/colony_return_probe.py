"""无食物、真实自留轨迹的预算返巢诊断，不训练或筛选检查点。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import math
from pathlib import Path
import subprocess

from pydantic import BaseModel
import torch

from colony_reward import Learner, RunConfig, WorldFrame, run_world


class Checkpoint(BaseModel):
    path: str
    sha256: str


class ProbeConfig(BaseModel):
    source_commit: str
    training_directory: str
    food_available: bool = False
    seeds: tuple[int, ...] = (9301, 9302, 9303, 9304)
    checkpoints: list[Checkpoint]
    training_config: RunConfig


class ReturnSummary(BaseModel):
    stage: str
    seed: int
    trails_visible: bool
    successful_individuals: int
    maximum_distance_before_first_return: list[float | None]


def summarize(directory: Path, stage: str, seed: int, visible: bool, ants: int) -> ReturnSummary:
    maximum = [0.] * ants
    first: list[float | None] = [None] * ants
    with gzip.open(directory / f"{stage}-{seed}-sample-True-trails-{visible}.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            frame = WorldFrame.model_validate_json(line)
            for ant in frame.ants:
                if first[ant.index] is None:
                    maximum[ant.index] = max(maximum[ant.index], math.hypot(ant.x, ant.y))
                    if ant.budget_return:
                        first[ant.index] = maximum[ant.index]
                if ant.picked_up or ant.delivered or any(ant.observation[0:72:8]):
                    raise AssertionError("无食物诊断出现了食物交互或来源信号")
    return ReturnSummary(stage=stage, seed=seed, trails_visible=visible,
                         successful_individuals=sum(value is not None for value in first),
                         maximum_distance_before_first_return=first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("logs/colony-reward/20260926T070617-2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = RunConfig.model_validate_json((args.training / "config.json").read_text())
    initial = args.training / "budget-000000.npz"
    budget = args.training / f"budget-{config.signal_updates:06d}.npz"
    final = [args.training / f"episode-{config.episodes:04d}-ant-{i:02d}.npz" for i in range(config.environment.ants)]
    paths = [initial, budget, *final]
    probe = ProbeConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        training_directory=str(args.training), training_config=config,
                        checkpoints=[Checkpoint(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])
    (args.output / "config.json").write_text(probe.model_dump_json(indent=2), encoding="utf-8")
    stages = (("initial", [initial] * config.environment.ants), ("budget", [budget] * config.environment.ants), ("reward", final))
    for stage, checkpoints in stages:
        for visible in ((True, False) if stage == "reward" else (True,)):
            for seed in probe.seeds:
                agents = [Learner(path, config, i, False) for i, path in enumerate(checkpoints)]
                run_world(agents, config, seed, stage, args.output, sampled=True, visible=visible, food_available=False)
                summary = summarize(args.output, stage, seed, visible, config.environment.ants)
                with (args.output / "returns.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(summary.model_dump_json() + "\n")
                print(summary.model_dump_json(), flush=True)
    for checkpoint in probe.checkpoints:
        if hashlib.sha256(Path(checkpoint.path).read_bytes()).hexdigest() != checkpoint.sha256:
            raise AssertionError("诊断改变了源模型文件")


if __name__ == "__main__":
    main()
