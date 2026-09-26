"""固定历史和信号，检查预算输入是否改变方向；不运行新世界。"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel
import torch

from colony_reward import WorldFrame
from mathhackson.training.foraging.policy import ForagingPolicy


class AuditConfig(BaseModel):
    source_commit: str
    traces: str
    checkpoints: str
    seeds: tuple[int, ...] = (9301, 9302, 9303, 9304)
    stages: tuple[str, ...] = ("initial", "budget", "reward")
    stride: int = 8


class InputArtifact(BaseModel):
    path: str
    sha256: str


class AntAudit(BaseModel):
    stage: str
    seed: int
    ant: int
    samples: int
    mean_budget_direction_change: float
    p90_budget_direction_change: float
    home_gradient_samples: int
    alignment_full_budget: float | None
    alignment_empty_budget: float | None
    alignment_difference: float | None
    maximum_distance_before_budget_zero: float
    moving_fraction_before_budget_zero: float | None
    first_budget_zero_tick: int | None
    hidden_reproduction_max_error: float


def checkpoint_for(root: Path, stage: str, ant: int) -> Path:
    if stage == "initial":
        return root / "budget-000000.npz"
    if stage == "budget":
        return root / "budget-001200.npz"
    return root / f"episode-0008-ant-{ant:02d}.npz"


def audit(config: AuditConfig, stage: str, seed: int, directory: Path) -> list[AntAudit]:
    models = [ForagingPolicy.load(checkpoint_for(Path(config.checkpoints), stage, i)) for i in range(8)]
    history: list[deque[torch.Tensor]] = [deque(maxlen=16) for _ in models]
    changes: list[list[float]] = [[] for _ in models]
    full_alignment: list[list[float]] = [[] for _ in models]
    empty_alignment: list[list[float]] = [[] for _ in models]
    maximum = [0.] * 8
    moves = [0] * 8
    steps = [0] * 8
    zero_tick: list[int | None] = [None] * 8
    hidden_error = [0.] * 8
    path = Path(config.traces) / f"{stage}-{seed}-sample-True-trails-True.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream, torch.inference_mode():
        for line in stream:
            frame = WorldFrame.model_validate_json(line)
            for ant in frame.ants:
                if not ant.active:
                    continue
                i = ant.index
                observation = torch.tensor(ant.observation)
                if zero_tick[i] is None:
                    if observation[76] == 0.:
                        zero_tick[i] = frame.tick
                    else:
                        maximum[i] = max(maximum[i], math.hypot(ant.x, ant.y))
                        moves[i] += int(ant.move)
                        steps[i] += 1
                _, actual_hidden, _ = models[i](observation, tuple(history[i]))
                error = float((actual_hidden - torch.tensor(ant.hidden)).abs().max())
                hidden_error[i] = max(hidden_error[i], error)
                if error > 1e-5:
                    raise AssertionError("模型、轨迹和隐藏状态不能重建一致")
                if frame.tick % config.stride == 0:
                    full, empty = observation.clone(), observation.clone()
                    full[76], empty[76] = 1., 0.
                    full[77] = empty[77] = 1.
                    a, _, _ = models[i](full, tuple(history[i]))
                    b, _, _ = models[i](empty, tuple(history[i]))
                    cosine = float((a * b).sum().clamp(-1., 1.))
                    changes[i].append(math.degrees(math.acos(cosine)))
                    local = torch.expm1(observation[:72].reshape(9, 8) * math.log(9.))[:, 1]
                    gradient = torch.stack((local[1] - local[3], local[2] - local[4]))
                    norm = float(gradient.norm())
                    if norm > .0001:
                        gradient /= norm
                        full_alignment[i].append(float(a @ gradient))
                        empty_alignment[i].append(float(b @ gradient))
                history[i].append(torch.tensor(ant.hidden))
    results = []
    for i in range(8):
        full = float(np.mean(full_alignment[i])) if full_alignment[i] else None
        empty = float(np.mean(empty_alignment[i])) if empty_alignment[i] else None
        results.append(AntAudit(stage=stage, seed=seed, ant=i, samples=len(changes[i]),
                               mean_budget_direction_change=float(np.mean(changes[i])),
                               p90_budget_direction_change=float(np.quantile(changes[i], .9)),
                               home_gradient_samples=len(full_alignment[i]), alignment_full_budget=full,
                               alignment_empty_budget=empty, alignment_difference=empty - full if empty is not None else None,
                               maximum_distance_before_budget_zero=maximum[i],
                               moving_fraction_before_budget_zero=moves[i] / steps[i] if steps[i] else None,
                               first_budget_zero_tick=zero_tick[i], hidden_reproduction_max_error=hidden_error[i]))
    with (directory / "ants.jsonl").open("a", encoding="utf-8") as stream:
        for result in results:
            stream.write(result.model_dump_json() + "\n")
            print(result.model_dump_json(), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = AuditConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         traces="logs/colony-return-probe/20260926T071059-2",
                         checkpoints="logs/colony-reward/20260926T070617-2")
    (args.output / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    paths = sorted(set(checkpoint_for(Path(config.checkpoints), stage, i) for stage in config.stages for i in range(8)))
    paths.extend(Path(config.traces) / f"{stage}-{seed}-sample-True-trails-True.jsonl.gz" for stage in config.stages for seed in config.seeds)
    inputs = [InputArtifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    with (args.output / "inputs.jsonl").open("x", encoding="utf-8") as stream:
        for item in inputs:
            stream.write(item.model_dump_json() + "\n")
    for stage in config.stages:
        for seed in config.seeds:
            audit(config, stage, seed, args.output)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in inputs):
        raise AssertionError("审计改变了输入产物")


if __name__ == "__main__":
    main()
