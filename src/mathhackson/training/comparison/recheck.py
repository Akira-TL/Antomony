"""不训练，只在修订信号环境中检查既有独立模型。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from .actors import NeuralForager
from .run import Config, SourceRecord, run_world


class RecheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_commit: str
    environment: ColonyConfig = ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.)
    evaluation_seeds: tuple[int, ...] = (9601, 9602, 9603, 9604)
    foundation_directory: str = "logs/comparator-foundations/20260926T083112-2"
    recurrent_directory: str = "logs/colony-reward/20260926T070617-2"
    motor_path: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    training_updates: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = RecheckConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    if args.smoke:
        config = config.model_copy(update={"evaluation_seeds": (9698,), "environment": ColonyConfig(
            ants=2, horizon=8, trail_profile="bounded-local-v2", nest_signal_strength=1.)})
    torch.set_num_threads(1)
    count = config.environment.ants
    base = Path(config.foundation_directory)
    signal_paths = [base / f"seed-{81 + i}-signal-002400.npz" for i in range(count)]
    reward_paths = [base / f"episode-0008-ant-{i:02d}.npz" for i in range(count)]
    recurrent_paths = [Path(config.recurrent_directory) / f"episode-0008-ant-{i:02d}.npz" for i in range(count)]
    sources = [Path(config.motor_path), *signal_paths, *reward_paths, *recurrent_paths]
    hashes = [SourceRecord(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    with (args.output / "sources.jsonl").open("x") as stream:
        stream.writelines(item.model_dump_json() + "\n" for item in hashes)
    motor, _ = load_motor(Path(config.motor_path))
    models = (("mlp-signal", [FeedforwardPolicy.load(path) for path in signal_paths]),
              ("mlp-reward", [FeedforwardPolicy.load(path) for path in reward_paths]),
              ("recurrent", [ForagingPolicy.load(path) for path in recurrent_paths]))
    run_config = Config(source_commit=config.source_commit, environment=config.environment,
                        signal_updates=0, episodes=0, evaluation_seeds=config.evaluation_seeds)
    for seed in config.evaluation_seeds:
        for task in ("food", "empty"):
            for label, policies in models:
                for sampled in (False, True):
                    actors = [NeuralForager(model, motor, seed * 32 + i) for i, model in enumerate(policies)]
                    result = run_world(actors, run_config, seed, label, task, sampled, args.output)
                    if any(result.updates):
                        raise AssertionError("冻结检查不能训练模型")
            rules = [LocalRuleController(seed * 32 + i) for i in range(count)]
            run_world(rules, run_config, seed, "rules", task, None, args.output)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in hashes):
        raise AssertionError("输入检查点文件改变")


if __name__ == "__main__":
    main()
