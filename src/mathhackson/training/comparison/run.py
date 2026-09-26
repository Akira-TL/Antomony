"""固定预算形成前馈对照；记录能力差异，不据此声称适应优势。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from .actors import NeuralForager, assert_reserved


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_commit: str
    model_seeds: tuple[int, ...] = tuple(range(81, 89))
    hidden_width: int = Field(default=14, gt=0, strict=True)
    signal_updates: int = 2400
    checkpoint_every: int = 200
    episodes: int = 8
    training_seed: int = 12000
    evaluation_seeds: tuple[int, ...] = (9601, 9602, 9603, 9604)
    environment: ColonyConfig = ColonyConfig()
    motor_path: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    recurrent_directory: str = "logs/colony-reward/20260926T070617-2"
    signal_rate: float = .003
    weight_decay: float = .0001
    signal_batch_size: int = 256


class SignalRecord(BaseModel):
    seed: int
    update: int
    loss: float
    mean_angle: float
    within_15: float


class WorldRecord(BaseModel):
    model: str
    seed: int
    sampled: bool | None
    task: str
    steps: int
    deliveries: int
    pickups: int
    budget_returns: int
    returning_individuals: int
    exhausted: int
    mean_max_radius: float
    updates: list[int]


class Frame(BaseModel):
    tick: int
    moves: list[bool]
    turns: list[float]
    positions: list[list[float]]
    carrying: list[bool]
    rewards: list[float]


class SourceRecord(BaseModel):
    path: str
    sha256: str


def train_signal(config: Config, directory: Path) -> list[FeedforwardPolicy]:
    models: list[FeedforwardPolicy] = []
    validation, target = signal_batch(np.random.default_rng(8301), 2048, budgets=True)
    for seed in config.model_seeds:
        model = FeedforwardPolicy(seed, hidden_width=config.hidden_width)
        rng = np.random.default_rng(seed + 10000)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config.signal_rate, weight_decay=config.weight_decay)
        model.save(directory / f"seed-{seed}-signal-000000.npz", update=0, phase="signal")
        for update in range(1, config.signal_updates + 1):
            inputs, expected = signal_batch(rng, config.signal_batch_size, budgets=True)
            loss = 1. - (model(inputs)[0] * expected).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            assert_reserved(model)
            if update % config.checkpoint_every == 0 or update == config.signal_updates:
                path = directory / f"seed-{seed}-signal-{update:06d}.npz"
                model.save(path, update=update, phase="signal")
                torch.save(optimizer.state_dict(), path.with_suffix(".optimizer.pt"))
                with torch.no_grad():
                    errors = torch.rad2deg(torch.acos((model(validation)[0] * target).sum(-1).clamp(-1., 1.)))
                record = SignalRecord(seed=seed, update=update, loss=float(loss.detach()),
                                      mean_angle=float(errors.mean()), within_15=float((errors < 15.).float().mean()))
                with (directory / "signal.jsonl").open("a") as output:
                    output.write(record.model_dump_json() + "\n")
        print(record.model_dump_json(), flush=True)
        models.append(model)
    return models


def run_world(actors: list[NeuralForager] | list[LocalRuleController], config: Config, seed: int,
              label: str, task: Literal["food", "empty"], sampled: bool | None, directory: Path) -> WorldRecord:
    env = ColonyEnvironment(seed, config.environment)
    if task == "empty":
        env.stock = 0
    max_radius = np.zeros(len(actors))
    returned = np.zeros(len(actors), dtype=np.bool_)
    before = [[p.detach().clone() for p in a.model.parameters()] if isinstance(a, NeuralForager) else [] for a in actors]
    for i, actor in enumerate(actors):
        if isinstance(actor, NeuralForager):
            actor.restart(seed * 32 + i)
    with gzip.open(directory / f"{label}-{task}-{seed}-{sampled}.jsonl.gz", "xt") as stream:
        while not env.done:
            active = [not ant.exhausted for ant in env.ants]
            actions = []
            for i, actor in enumerate(actors):
                obs = env.observation(i)
                action = (actor.act(obs, sampled=bool(sampled)) if isinstance(actor, NeuralForager) else actor.act(obs)) if active[i] else DirectionAction(False, 0., 0.)
                actions.append(action)
            events = env.step(actions)
            for i, actor in enumerate(actors):
                max_radius[i] = max(max_radius[i], float(np.linalg.norm(env.ants[i].position)))
                returned[i] |= events[i].budget_return
                if active[i] and isinstance(actor, NeuralForager):
                    actor.feedback(events[i].reward, env.observation(i), terminal=env.done or events[i].exhausted)
            stream.write(Frame(tick=env.steps, moves=[a.move for a in actions], turns=[a.turn for a in actions],
                               positions=[a.position.tolist() for a in env.ants], carrying=[a.carrying for a in env.ants],
                               rewards=[e.reward for e in events]).model_dump_json() + "\n")
    for actor, weights in zip(actors, before, strict=True):
        if isinstance(actor, NeuralForager) and actor.optimizer is None:
            if any(not a.equal(b) for a, b in zip(actor.model.parameters(), weights, strict=True)):
                raise AssertionError("能力检查期间参数改变")
    result = WorldRecord(model=label, seed=seed, sampled=sampled, task=task, steps=env.steps,
                         deliveries=sum(a.deliveries for a in env.ants), pickups=sum(a.pickups for a in env.ants),
                         budget_returns=sum(a.budget_returns for a in env.ants), returning_individuals=int(returned.sum()),
                         exhausted=sum(a.exhausted for a in env.ants), mean_max_radius=float(max_radius.mean()),
                         updates=[a.updates if isinstance(a, NeuralForager) else 0 for a in actors])
    with (directory / "worlds.jsonl").open("a") as output:
        output.write(result.model_dump_json() + "\n")
    print(result.model_dump_json(), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Config(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    if args.smoke:
        config = config.model_copy(update={"model_seeds": (81, 82), "signal_updates": 2, "episodes": 1,
                                           "evaluation_seeds": (9699,), "environment": ColonyConfig(ants=2, horizon=8)})
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    sources = [Path(config.motor_path), *[Path(config.recurrent_directory) / f"episode-0008-ant-{i:02d}.npz" for i in range(config.environment.ants)]]
    hashes = [SourceRecord(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    with (args.output / "sources.jsonl").open("x") as output:
        output.writelines(item.model_dump_json() + "\n" for item in hashes)
    motor, _ = load_motor(Path(config.motor_path))
    motor_before = [p.detach().clone() for p in motor.parameters()]
    models = train_signal(config, args.output)
    actors = [NeuralForager(model, motor, seed, training=True) for model, seed in zip(models, config.model_seeds, strict=True)]
    for episode in range(1, config.episodes + 1):
        run_world(actors, config, config.training_seed + episode, "training", "food", True, args.output)
        if episode % 2 == 0 or episode == config.episodes:
            for i, actor in enumerate(actors):
                path = args.output / f"episode-{episode:04d}-ant-{i:02d}.npz"
                actor.model.save(path, update=actor.updates, phase="reward")
                torch.save(actor.optimizer.state_dict(), path.with_suffix(".optimizer.pt"))
    recurrent = [ForagingPolicy.load(path) for path in sources[1:]]
    for seed in config.evaluation_seeds:
        for task in ("food", "empty"):
            for label, policies in (("mlp-signal", models), ("mlp-reward", [a.model for a in actors]), ("recurrent", recurrent)):
                for sampled in (False, True):
                    controls = [NeuralForager(model, motor, seed * 32 + i) for i, model in enumerate(policies)]
                    run_world(controls, config, seed, label, task, sampled, args.output)
            rules = [LocalRuleController(seed * 32 + i) for i in range(config.environment.ants)]
            run_world(rules, config, seed, "rules", task, None, args.output)
    if any(not a.equal(b) for a, b in zip(motor_before, motor.parameters(), strict=True)):
        raise AssertionError("冻结动作底座被改变")
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in hashes):
        raise AssertionError("输入检查点文件改变")


if __name__ == "__main__":
    main()
