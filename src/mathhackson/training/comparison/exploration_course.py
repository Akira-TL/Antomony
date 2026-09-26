"""有预算的无食物探索课程，保留旧参数，固定最终轮次评价。"""
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
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from .actors import NeuralForager, assert_reserved
from .run import Config, SourceRecord, run_world


class CoursePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    training_seeds: tuple[int, ...] = tuple(range(15201, 15233))
    evaluation_seeds: tuple[int, ...] = tuple(range(15101, 15109))
    checkpoint_every: int = Field(default=4, ge=1)
    food_every: int = Field(default=4, ge=2)
    empty_exploration_reward: float = Field(default=.05, gt=0.)
    environment: ColonyConfig = ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.)
    motor: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    recurrent_directory: str = "logs/colony-reward/20260926T070617-2"
    mlp_directory: str = "logs/comparator-foundations/20260926T083112-2"

    @model_validator(mode="after")
    def separate_seeds(self) -> CoursePlan:
        if not self.training_seeds or not self.evaluation_seeds:
            raise ValueError("训练和评价种子均不能为空")
        if any(len(set(s)) != len(s) for s in (self.training_seeds, self.evaluation_seeds)):
            raise ValueError("种子不能重复")
        if set(self.training_seeds) & set(self.evaluation_seeds):
            raise ValueError("训练与评价种子必须分离")
        return self


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    smoke: bool
    snapshot_episodes: list[int] = []
    final_updates: list[int] = []


def checkpoint(actors: list[NeuralForager], directory: Path, episode: int) -> None:
    for i, actor in enumerate(actors):
        path = directory / f"episode-{episode:04d}-ant-{i:02d}.npz"
        assert_reserved(actor.model)
        actor.model.save(path, update=actor.updates, phase="reward")
        with path.with_suffix(".optimizer.pt").open("xb") as stream:
            torch.save(actor.optimizer.state_dict(), stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = CoursePlan.model_validate_json(args.config.read_text())
    if args.smoke:
        plan = plan.model_copy(update={"training_seeds": (15991, 15992), "evaluation_seeds": (15999,),
            "checkpoint_every": 1, "food_every": 2, "environment": plan.environment.model_copy(update={"ants": 2, "horizon": 8})})
    count = plan.environment.ants
    paths = [Path(plan.motor), *[Path(plan.recurrent_directory) / f"episode-0008-ant-{i:02d}.npz" for i in range(count)],
             *[Path(plan.mlp_directory) / f"seed-{81 + i}-signal-002400.npz" for i in range(count)]]
    sources = [SourceRecord(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    original = [ForagingPolicy.load(path) for path in paths[1:count + 1]]
    mlp = [FeedforwardPolicy.load(path) for path in paths[count + 1:]]
    motor, _ = load_motor(Path(plan.motor))
    torch.set_num_threads(1)
    actors = [NeuralForager(model, motor, 17000 + i, training=True) for i, model in enumerate(original)]
    motor_before = [[p.detach().clone() for p in a.motor.parameters()] for a in actors]
    execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          started_at=datetime.now(timezone.utc).isoformat(), smoke=args.smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    training, evaluation, snapshots = [args.output / name for name in ("training", "evaluation", "snapshots")]
    for path in (training, evaluation, snapshots):
        path.mkdir()
    (args.output / "config.json").write_text(plan.model_dump_json(indent=2))
    (args.output / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (args.output / "sources.jsonl").open("x") as stream:
        stream.writelines(s.model_dump_json() + "\n" for s in sources)
    checkpoint(actors, snapshots, 0)
    execution.snapshot_episodes.append(0)
    for episode, seed in enumerate(plan.training_seeds, 1):
        task = "food" if episode % plan.food_every == 0 else "empty"
        environment = plan.environment if task == "food" else plan.environment.model_copy(
            update={"exploration_reward": plan.empty_exploration_reward})
        config = Config(source_commit=execution.source_commit, environment=environment, signal_updates=0, episodes=0)
        run_world(actors, config, seed, "training", task, True, training)
        if episode % plan.checkpoint_every == 0 or episode == len(plan.training_seeds):
            checkpoint(actors, snapshots, episode)
            execution.snapshot_episodes.append(episode)
    for actor, before in zip(actors, motor_before, strict=True):
        if any(not a.equal(b) for a, b in zip(actor.motor.parameters(), before, strict=True)):
            raise AssertionError("课程改变了动作底座")
    execution.final_updates = [actor.updates for actor in actors]
    final = [ForagingPolicy.load(snapshots / f"episode-{len(plan.training_seeds):04d}-ant-{i:02d}.npz") for i in range(count)]
    config = Config(source_commit=execution.source_commit, environment=plan.environment, signal_updates=0, episodes=0)
    for seed in plan.evaluation_seeds:
        for task in ("food", "empty"):
            for name, models in (("recurrent-before", original), ("recurrent-after", final), ("mlp", mlp)):
                frozen = [NeuralForager(model, motor, seed * 32 + i) for i, model in enumerate(models)]
                result = run_world(frozen, config, seed, name, task, True, evaluation)
                if any(result.updates):
                    raise AssertionError("评价不能继续训练")
            run_world([LocalRuleController(seed * 32 + i) for i in range(count)], config, seed,
                      "rules", task, None, evaluation)
    if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in sources):
        raise AssertionError("原检查点文件改变")
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (args.output / "execution.json").write_text(execution.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
