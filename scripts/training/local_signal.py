"""训练八接收器基础信号映射，并在无全局答案的真实往返环境留出诊断。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel
import torch

from direction_adaptation import FOUNDATION
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.environment import ForagingConfig, ForagingEnvironment
from mathhackson.training.foraging.policy import ForagingController, ForagingPolicy


class TrainingConfig(BaseModel):
    source_commit: str
    motor_path: str
    motor_sha256: str
    policy_seed: int = 71
    updates: int = 1200
    batch_size: int = 256
    learning_rate: float = .003
    checkpoint_every: int = 100
    validation_seed: int = 8101
    validation_size: int = 2048
    evaluation_seeds: tuple[int, ...] = tuple(range(9001, 9013))
    environment: ForagingConfig = ForagingConfig()


class SignalMetrics(BaseModel):
    mean_angle_degrees: float
    p90_angle_degrees: float
    within_15_degrees: float


class TrainingRecord(BaseModel):
    update: int
    loss: float
    validation: SignalMetrics


class ForagingFrame(BaseModel):
    tick: int
    receptors: list[list[float]]
    hidden: list[float]
    direction: list[float]
    carrying_before: bool
    x: float
    y: float
    heading: float
    move: bool
    turn: float
    picked_up: bool
    delivered: bool
    reward: float


class Trial(BaseModel):
    seed: int
    trails_visible: bool
    steps: int
    pickups: int
    deliveries: int
    total_reward: float
    motor_unchanged: bool
    policy_unchanged: bool
    reserved_connections_zero: bool


def signal_metrics(model: ForagingPolicy, config: TrainingConfig) -> SignalMetrics:
    inputs, targets = signal_batch(np.random.default_rng(config.validation_seed), config.validation_size)
    with torch.inference_mode():
        prediction, _, _ = model(inputs)
        error = torch.rad2deg(torch.acos((prediction * targets).sum(-1).clamp(-1., 1.))).numpy()
    return SignalMetrics(mean_angle_degrees=float(error.mean()), p90_angle_degrees=float(np.quantile(error, .9)),
                         within_15_degrees=float(np.mean(error <= 15.)))


def train(config: TrainingConfig, directory: Path) -> Path:
    model = ForagingPolicy(config.policy_seed)
    rng = np.random.default_rng(config.policy_seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate, weight_decay=.0001)
    model.save(directory / "signal-000000.npz", update=0, phase="signal")
    for update in range(1, config.updates + 1):
        inputs, target = signal_batch(rng, config.batch_size)
        prediction, _, _ = model(inputs)
        loss = 1. - (prediction * target).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if any(bool(p.any()) or p.grad is not None for p in model.reserved_parameters()):
            raise AssertionError("预留接收器在基础课程中被训练")
        if update % config.checkpoint_every == 0:
            model.save(directory / f"signal-{update:06d}.npz", update=update, phase="signal")
            record = TrainingRecord(update=update, loss=float(loss.detach()), validation=signal_metrics(model, config))
            with (directory / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
            print(record.model_dump_json(), flush=True)
    return directory / f"signal-{config.updates:06d}.npz"


def evaluate(checkpoint: Path, config: TrainingConfig, seed: int, visible: bool, directory: Path) -> Trial:
    model = ForagingPolicy.load(checkpoint)
    motor, _ = load_motor(Path(config.motor_path))
    motor_before = [p.detach().clone() for p in motor.parameters()]
    policy_before = [p.detach().clone() for p in model.parameters()]
    controller = ForagingController(model, motor)
    env = ForagingEnvironment(seed, config.environment)
    total_reward = 0.
    with (directory / f"scene-{seed}-trails-{visible}.jsonl").open("x", encoding="utf-8") as stream:
        while not env.done:
            observed = env.observation()
            if not visible:
                receptors = observed.receptors.copy()
                receptors[:, 1:3] = 0.
                observed = replace(observed, receptors=receptors)
            decision = controller.decide(observed)
            outcome = env.step(decision.action.move, decision.action.turn)
            total_reward += outcome.reward
            frame = ForagingFrame(tick=env.steps, receptors=observed.receptors.tolist(),
                                  hidden=controller.history[-1].tolist(), direction=decision.direction.tolist(),
                                  carrying_before=observed.carrying, x=float(env.position[0]), y=float(env.position[1]),
                                  heading=env.heading, move=decision.action.move, turn=decision.action.turn,
                                  picked_up=outcome.picked_up, delivered=outcome.delivered, reward=outcome.reward)
            stream.write(frame.model_dump_json() + "\n")
    motor_same = all(a.equal(b) for a, b in zip(motor_before, motor.parameters(), strict=True))
    policy_same = all(a.equal(b) for a, b in zip(policy_before, model.parameters(), strict=True))
    reserved_zero = not any(bool(p.any()) for p in model.reserved_parameters())
    if not motor_same or not policy_same or not reserved_zero:
        raise AssertionError("评估期间权重改变或预留连接非零")
    return Trial(seed=seed, trails_visible=visible, steps=env.steps, pickups=env.pickups, deliveries=env.deliveries,
                 total_reward=total_reward, motor_unchanged=motor_same, policy_unchanged=policy_same,
                 reserved_connections_zero=reserved_zero)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    directory = parser.parse_args().output
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    motor_path = FOUNDATION / "seed-41/update-001200.npz"
    config = TrainingConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                            motor_path=str(motor_path), motor_sha256=hashlib.sha256(motor_path.read_bytes()).hexdigest())
    (directory / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    checkpoint = train(config, directory)
    for seed in config.evaluation_seeds:
        for visible in (True, False):
            result = evaluate(checkpoint, config, seed, visible, directory)
            with (directory / "trials.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(result.model_dump_json() + "\n")
            print(result.model_dump_json(), flush=True)
    if hashlib.sha256(motor_path.read_bytes()).hexdigest() != config.motor_sha256:
        raise AssertionError("基础检查点文件发生变化")


if __name__ == "__main__":
    main()
