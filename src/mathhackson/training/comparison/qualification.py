"""冻结模型的基础能力检查，不在检查世界中训练或挑选参数。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

from pydantic import BaseModel, ConfigDict, Field
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from .actors import NeuralForager
from .run import Config, SourceRecord, run_world


class QualificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = tuple(range(14101, 14109))
    environment: ColonyConfig = ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.)
    motor: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    mlp_directory: str = "logs/comparator-foundations/20260926T083112-2"
    recurrent_directory: str = "logs/colony-reward/20260926T070617-2"
    minimum_delivery_fraction: float = Field(default=.9, ge=0., le=1.)
    minimum_returning_fraction: float = Field(default=.5, ge=0., le=1.)
    minimum_exploration_radius: float = Field(default=4.5, gt=0.)


class ModelSize(BaseModel):
    model: str
    policy_parameters: int
    motor_parameters: int
    history_steps: int
    history_width: int


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    training_updates: int = 0
    smoke: bool
    models: list[ModelSize]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = QualificationPlan.model_validate_json(args.config.read_text())
    if len(set(plan.seeds)) != len(plan.seeds) or not plan.seeds:
        raise ValueError("世界种子须非空且不重复")
    if args.smoke:
        plan = plan.model_copy(update={"seeds": (14999,), "environment": plan.environment.model_copy(
            update={"ants": 2, "horizon": 8})})
    count = plan.environment.ants
    mlp_paths = [Path(plan.mlp_directory) / f"seed-{81 + i}-signal-002400.npz" for i in range(count)]
    recurrent_paths = [Path(plan.recurrent_directory) / f"episode-0008-ant-{i:02d}.npz" for i in range(count)]
    paths = [Path(plan.motor), *mlp_paths, *recurrent_paths]
    sources = [SourceRecord(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    torch.set_num_threads(1)
    motor, _ = load_motor(Path(plan.motor))
    models = (("mlp", [FeedforwardPolicy.load(path) for path in mlp_paths]),
              ("recurrent", [ForagingPolicy.load(path) for path in recurrent_paths]))
    execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          started_at=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
                          models=[ModelSize(model=name, policy_parameters=sum(p.numel() for p in policies[0].parameters()),
                                            motor_parameters=sum(p.numel() for p in motor.parameters()),
                                            history_steps=16 if name == "recurrent" else 0,
                                            history_width=8 if name == "recurrent" else 0) for name, policies in models])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(plan.model_dump_json(indent=2))
    (args.output / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (args.output / "sources.jsonl").open("x") as stream:
        stream.writelines(item.model_dump_json() + "\n" for item in sources)
    config = Config(source_commit=execution.source_commit, environment=plan.environment,
                    signal_updates=0, episodes=0, evaluation_seeds=plan.seeds)
    for seed in plan.seeds:
        for task in ("food", "empty"):
            for name, policies in models:
                actors = [NeuralForager(model, motor, seed * 32 + i) for i, model in enumerate(policies)]
                result = run_world(actors, config, seed, name, task, True, args.output)
                if any(result.updates):
                    raise AssertionError("基础检查不得修改模型")
            rules = [LocalRuleController(seed * 32 + i) for i in range(count)]
            run_world(rules, config, seed, "rules", task, None, args.output)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in sources):
        raise AssertionError("输入参数文件被修改")
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (args.output / "execution.json").write_text(execution.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
