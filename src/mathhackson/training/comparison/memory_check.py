"""转换既有MLP并核对零记忆连接的逐帧等价；不是效能实验。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import subprocess

from pydantic import BaseModel
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from .actors import NeuralForager
from .run import Config, SourceRecord, run_world


class EqualityRecord(BaseModel):
    seed: int
    task: str
    frames: int
    trajectory_sha256: str
    original_parameters: int
    memory_parameters: int
    motor_parameters: int


def fingerprint(path: Path) -> SourceRecord:
    return SourceRecord(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = Config(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    evaluation_seeds=(14101, 14102),
                    environment=ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    # 工作树可能尚未提交；另外保存实际代码散列，不能只用HEAD冒充执行代码身份。
    sources = [Path(config.motor_path), *[Path("logs/comparator-foundations/20260926T083112-2") /
               f"seed-{seed}-signal-002400.npz" for seed in config.model_seeds]]
    hashes = [fingerprint(path) for path in [*sources, *sorted(Path("src/mathhackson").rglob("*.py")),
                                           Path("scripts/training/memory-check.sh")]]
    with (args.output / "sources.jsonl").open("x") as stream:
        stream.writelines(item.model_dump_json() + "\n" for item in hashes)
    motor, _ = load_motor(sources[0])
    original = [FeedforwardPolicy.load(path) for path in sources[1:]]
    memories = []
    for i, base in enumerate(original):
        path = args.output / f"episode-0000-ant-{i:02d}.npz"
        MemoryPolicy(base).save(path, update=0, phase="frozen")
        memories.append(MemoryPolicy.load(path))
    for seed in config.evaluation_seeds:
        for task in ("food", "empty"):
            records, traces = [], []
            for label, models in (("mlp", original), ("memory", memories)):
                actors = [NeuralForager(model, motor, seed * 32 + i) for i, model in enumerate(models)]
                records.append(run_world(actors, config, seed, label, task, True, args.output))
                with gzip.open(args.output / f"{label}-{task}-{seed}-True.jsonl.gz", "rb") as stream:
                    traces.append(stream.read())
                if label == "memory" and any(len(actor.history) != 16 for actor in actors):
                    raise AssertionError("个体记忆未保留完整时间范围")
            if records[0].model_dump(exclude={"model"}) != records[1].model_dump(exclude={"model"}):
                raise AssertionError("转换后世界统计不等价")
            if traces[0] != traces[1]:
                raise AssertionError("转换后逐帧轨迹不等价")
            record = EqualityRecord(seed=seed, task=task, frames=len(traces[0].splitlines()),
                                    trajectory_sha256=hashlib.sha256(traces[0]).hexdigest(),
                                    original_parameters=sum(p.numel() for p in original[0].parameters()),
                                    memory_parameters=sum(p.numel() for p in memories[0].parameters()),
                                    motor_parameters=sum(p.numel() for p in motor.parameters()))
            with (args.output / "equality.jsonl").open("a") as stream:
                stream.write(record.model_dump_json() + "\n")
    if any(fingerprint(Path(item.path)) != item for item in hashes):
        raise AssertionError("输入检查点或执行代码在检查期间改变")
    with (args.output / "outputs.jsonl").open("x") as stream:
        stream.writelines(fingerprint(path).model_dump_json() + "\n" for path in sorted(args.output.iterdir())
                          if path.name != "outputs.jsonl")


if __name__ == "__main__":
    main()
