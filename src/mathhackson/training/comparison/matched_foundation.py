"""固定预算训练近似规模MLP，再冻结参数检查基础能力。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from .actors import NeuralForager
from .run import Config, SourceRecord, run_world, train_signal


class MatchedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_seeds: tuple[int, ...] = tuple(range(81, 89))
    evaluation_seeds: tuple[int, ...] = tuple(range(17101, 17109))
    hidden_width: int = Field(default=17, gt=0, strict=True)
    signal_updates: int = Field(default=2400, gt=0)
    checkpoint_every: int = Field(default=200, gt=0)
    signal_batch_size: int = Field(default=256, gt=0)
    signal_rate: float = Field(default=.003, gt=0., allow_inf_nan=False)
    weight_decay: float = Field(default=.0001, ge=0., allow_inf_nan=False)
    environment: ColonyConfig = ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.)
    motor: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    original_directory: str = "logs/comparator-foundations/20260926T083112-2"
    minimum_delivery_fraction: float = Field(default=.9, ge=0., le=1.)
    minimum_returning_fraction: float = Field(default=.5, ge=0., le=1.)
    minimum_exploration_radius: float = Field(default=4.5, gt=0.)

    @model_validator(mode="after")
    def validate_units(self) -> MatchedPlan:
        if len(self.model_seeds) != self.environment.ants or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("模型种子须独立且与蚂蚁数一致")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("评价种子须非空且唯一")
        return self


class ModelSize(BaseModel):
    model: str
    policy_parameters: int
    motor_parameters: int
    reserved_connections: int
    frozen_value_parameters: int = 9


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    smoke: bool
    evaluation_updates: int = 0
    training_updates_per_individual: int
    models: list[ModelSize]


def fingerprint(path: Path) -> SourceRecord:
    return SourceRecord(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(plan: MatchedPlan, directory: Path, *, smoke: bool = False) -> None:
    if smoke:
        plan = MatchedPlan.model_validate({**plan.model_dump(), "model_seeds": plan.model_seeds[:2],
            "evaluation_seeds": (17999,), "signal_updates": 2, "checkpoint_every": 1,
            "environment": plan.environment.model_copy(update={"ants": 2, "horizon": 8})})
    paths = [Path(plan.motor), *[Path(plan.original_directory) / f"seed-{seed}-signal-002400.npz"
                               for seed in plan.model_seeds]]
    sources = [fingerprint(path) for path in paths]
    torch.set_num_threads(1)
    motor, _ = load_motor(paths[0])
    originals = [FeedforwardPolicy.load(path) for path in paths[1:]]
    if any(model.hidden_width != 14 for model in originals):
        raise ValueError("参照必须是原14宽模型")
    directory.mkdir(parents=True, exist_ok=False)
    execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          started_at=datetime.now(timezone.utc).isoformat(), smoke=smoke,
                          training_updates_per_individual=plan.signal_updates, models=[])
    (directory / "config.json").write_text(plan.model_dump_json(indent=2))
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (directory / "sources.jsonl").open("x") as stream:
        stream.writelines(item.model_dump_json() + "\n" for item in sources)
    config = Config(source_commit=execution.source_commit, model_seeds=plan.model_seeds,
                    hidden_width=plan.hidden_width, signal_updates=plan.signal_updates,
                    checkpoint_every=plan.checkpoint_every, signal_batch_size=plan.signal_batch_size,
                    signal_rate=plan.signal_rate, weight_decay=plan.weight_decay,
                    episodes=0, evaluation_seeds=plan.evaluation_seeds, environment=plan.environment)
    train_signal(config, directory)
    final_paths = [directory / f"seed-{seed}-signal-{plan.signal_updates:06d}.npz" for seed in plan.model_seeds]
    final_sources = [fingerprint(path) for path in final_paths]
    models = (("mlp-matched", [FeedforwardPolicy.load(path) for path in final_paths]), ("mlp-original", originals))
    for label, policies in models:
        for policy in policies:
            policy.assert_reserved()
        execution.models.append(ModelSize(model=label, policy_parameters=sum(p.numel() for p in policies[0].parameters()),
            motor_parameters=sum(p.numel() for p in motor.parameters()), reserved_connections=50 * policies[0].hidden_width))
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))
    for seed in plan.evaluation_seeds:
        for task in ("food", "empty"):
            for label, policies in models:
                actors = [NeuralForager(policy, motor, seed * 32 + i) for i, policy in enumerate(policies)]
                result = run_world(actors, config, seed, label, task, True, directory)
                if any(result.updates):
                    raise AssertionError("评价期间不得更新")
            rules = [LocalRuleController(seed * 32 + i) for i in range(plan.environment.ants)]
            run_world(rules, config, seed, "rules", task, None, directory)
    if any(fingerprint(Path(item.path)).sha256 != item.sha256 for item in sources + final_sources):
        raise AssertionError("源参数或最终参数文件被修改")
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(MatchedPlan.model_validate_json(args.config.read_text()), args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
