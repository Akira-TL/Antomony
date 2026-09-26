"""在固定训练条件上学习接受更新，再运行独立方向序列的开发对照。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel
import torch

from direction_adaptation import FOUNDATION, make_optimizer
from mathhackson.training.direction.adaptation import DirectionCorrection, local_direction_loss
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.environment import DirectionEnvironment, MAX_TURN, STEP_DISTANCE, wrap_angle
from mathhackson.training.direction.gate import Feedback, FeedbackHistory, UpdateGate, accept_proposal

Scenario = Literal["normal", "positive", "negative", "transient", "periodic"]
Mode = Literal["off", "always", "rule", "learned", "random"]
SCENARIOS: tuple[Scenario, ...] = ("normal", "positive", "negative", "transient", "periodic")
HORIZON = 192
TEACHER_STEPS = 8
WRITE_COST = .00001


class Episode(BaseModel):
    motor_seed: int
    scene_seed: int
    scenario: Scenario
    magnitude: float
    period: int = 48

    def bias(self, tick: int) -> float:
        if tick < 64 or self.scenario == "normal":
            return 0.
        if self.scenario == "positive":
            return self.magnitude
        if self.scenario == "negative":
            return -self.magnitude
        if self.scenario == "transient":
            return self.magnitude if tick < 76 else 0.
        return self.magnitude * math.sin(2. * math.pi * (tick - 64) / self.period)

    def commands(self) -> np.ndarray:
        return np.random.default_rng(self.scene_seed).uniform(-math.pi, math.pi, 5)


class Frame(BaseModel):
    tick: int
    desired: float
    heading: float
    x: float
    y: float
    move: bool
    turn: float
    error_degrees: float
    progress_fraction: float
    offset_before: float
    proposed: float
    offset_after: float
    accept: bool
    probability: float


class Result(BaseModel):
    episode: Episode
    mode: Mode
    gate_seed: int | None
    mean_error_degrees: float
    progress_fraction: float
    updates: int
    accepts: int
    motor_unchanged: bool
    error_difference: float = 0.
    progress_difference: float = 0.


class Config(BaseModel):
    source_commit: str
    purpose: str = "学习更新判断的工程开发诊断，不是正式研究结论"
    train: list[Episode]
    evaluation: list[Episode]
    gate_seeds: tuple[int, ...] = (101, 102, 103)
    updates: int = 600
    batch: int = 128
    learning_rate: float = .003
    teacher_steps: int = TEACHER_STEPS
    write_cost: float = WRITE_COST


@dataclass
class Individual:
    model: DirectionCorrection
    optimizer: torch.optim.Optimizer
    environment: DirectionEnvironment
    history: FeedbackHistory
    original: list[torch.Tensor]

    @classmethod
    def create(cls, episode: Episode) -> Individual:
        motor, _ = load_motor(FOUNDATION / f"seed-{episode.motor_seed}/update-001200.npz")
        model = DirectionCorrection(motor)
        optimizer = make_optimizer(model, "mulo")
        assert optimizer is not None
        return cls(model, optimizer, DirectionEnvironment(0., float(episode.commands()[0])),
                   FeedbackHistory(), [p.detach().clone() for p in motor.parameters()])

    def observe_and_propose(self, desired: float, bias: float, tick: int) -> tuple[Frame, np.ndarray]:
        env, model = self.environment, self.model
        env.desired = desired
        before = float(model.offset.detach()[0])
        heading_before = env.heading
        action = model.command(env.observation())
        turn = float(action.turn.detach())
        progress = env.step(action.move, turn, execution_bias=bias)
        error = wrap_angle(desired - env.heading)
        residual = wrap_angle(env.heading - heading_before) - MAX_TURN * turn
        self.history.append(Feedback(error, turn, residual))
        self.optimizer.zero_grad()
        local_direction_loss(action.turn, error).backward()
        self.optimizer.step()
        model.project()
        proposed = float(model.offset.detach()[0])
        # 优化器统计已消费本次反馈，但行为参数必须等待接受判断。
        accept_proposal(model, before, True)
        frame = Frame(tick=tick, desired=desired, heading=env.heading, x=env.x, y=env.y,
                      move=action.move, turn=turn, error_degrees=abs(math.degrees(error)),
                      progress_fraction=progress / STEP_DISTANCE, offset_before=before,
                      proposed=proposed, offset_after=before, accept=False, probability=0.)
        return frame, self.history.features(before, proposed)

    def assert_frozen(self) -> None:
        if not all(a.equal(b) for a, b in zip(self.original, self.model.motor.parameters(), strict=True)):
            raise AssertionError("基础动作参数被修改")


def teacher_benefit(individual: Individual, episode: Episode, tick: int, proposed: float) -> float:
    """仅采样训练标签时调用；不允许在留出轨迹运行时调用。"""
    model = individual.model
    before = float(model.offset.detach()[0])
    commands = episode.commands()
    costs: list[float] = []
    try:
        with torch.no_grad():
            for offset in (before, proposed):
                accept_proposal(model, offset, True)
                env = replace(individual.environment)
                errors: list[float] = []
                for future in range(tick + 1, tick + 1 + TEACHER_STEPS):
                    env.desired = float(commands[future // 48])
                    action = model.command(env.observation())
                    env.step(action.move, float(action.turn), execution_bias=episode.bias(future))
                    errors.append(abs(wrap_angle(env.desired - env.heading)))
                costs.append(float(np.mean(errors)))
    finally:
        accept_proposal(model, before, True)
    return costs[0] - costs[1] - WRITE_COST


def collect(config: Config, directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[np.ndarray] = []
    benefits: list[float] = []
    identities: list[tuple[int, int]] = []
    for episode_index, episode in enumerate(config.train):
        individual = Individual.create(episode)
        rng = np.random.default_rng(episode.scene_seed + 9000)
        commands = episode.commands()
        for tick in range(HORIZON):
            frame, observed = individual.observe_and_propose(float(commands[tick // 48]), episode.bias(tick), tick)
            if tick % 4 == 0:
                benefits.append(teacher_benefit(individual, episode, tick, frame.proposed))
                features.append(observed)
                identities.append((episode_index, tick))
            accept_proposal(individual.model, frame.proposed, bool(rng.random() < .5))
        individual.assert_frozen()
        print(f"采样 {episode_index + 1}/{len(config.train)}", flush=True)
    x, benefit = np.stack(features), np.asarray(benefits, dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(benefit).all():
        raise ValueError("教师样本非有限")
    with (directory / "training-data.npz").open("xb") as stream:
        np.savez(stream, features=x, benefit=benefit, identities=np.asarray(identities))
    print(f"教师接受比例 {float(np.mean(benefit > 0.)):.4f}", flush=True)
    return torch.from_numpy(x), torch.from_numpy(benefit)


def train_gate(config: Config, x: torch.Tensor, benefit: torch.Tensor, seed: int, directory: Path) -> UpdateGate:
    model = UpdateGate(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    labels = (benefit > 0.).float()
    importance = (benefit.abs() / .005).clamp(.05, 10.)
    path = directory / f"gate-{seed}"
    path.mkdir()
    model.save(path / "update-000000.npz", update=0)
    for update in range(1, config.updates + 1):
        index = torch.randint(len(x), (config.batch,), generator=generator)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(model(x[index]), labels[index], reduction="none")
        loss = (losses * importance[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % 100 == 0:
            model.save(path / f"update-{update:06d}.npz", update=update)
            with torch.no_grad():
                predictions = model(x) >= 0.
                accuracy = float((predictions == labels.bool()).float().mean())
                positive = float(predictions.float().mean())
            print(f"判断训练 seed={seed} step={update} loss={float(loss.detach()):.6f} accuracy={accuracy:.4f} accepts={positive:.4f}", flush=True)
    return UpdateGate.load(path / f"update-{config.updates:06d}.npz")


def evaluate(episode: Episode, mode: Mode, gate_seed: int | None, gate: UpdateGate | None,
             directory: Path, *, schedule: np.ndarray | None = None) -> Result:
    if mode == "random":
        if schedule is None or schedule.shape != (HORIZON,) or schedule.dtype != np.bool_:
            raise ValueError("随机接受时刻表必须包含每步的布尔选择")
    elif schedule is not None:
        raise ValueError("其他策略不能读取随机接受时刻表")
    individual = Individual.create(episode)
    commands = episode.commands()
    frames: list[Frame] = []
    for tick in range(HORIZON):
        frame, observed = individual.observe_and_propose(float(commands[tick // 48]), episode.bias(tick), tick)
        if mode == "always":
            probability = 1.
        elif mode == "random":
            assert schedule is not None
            probability = float(schedule[tick])
        elif mode == "learned":
            assert gate is not None
            probability = gate.probability(observed)
        elif mode == "rule":
            error = wrap_angle(frame.desired - frame.heading)
            probability = float(abs(error) > math.radians(1.) and (frame.proposed - frame.offset_before) * error > 0.)
        else:
            probability = 0.
        accepted = probability >= .5
        accept_proposal(individual.model, frame.proposed, accepted)
        frames.append(frame.model_copy(update={"accept": accepted, "probability": probability,
                                              "offset_after": float(individual.model.offset.detach()[0])}))
        if (tick + 1) % 32 == 0:
            with (directory / f"step-{tick + 1:03d}.pt").open("xb") as stream:
                torch.save({"offset": individual.model.offset.detach().clone(),
                            "optimizer": individual.optimizer.state_dict(), "tick": tick}, stream)
    individual.assert_frozen()
    with (directory / "frames.jsonl").open("x", encoding="utf-8") as stream:
        stream.writelines(frame.model_dump_json() + "\n" for frame in frames)
    return Result(episode=episode, mode=mode, gate_seed=gate_seed,
                  mean_error_degrees=float(np.mean([f.error_degrees for f in frames[64:]])),
                  progress_fraction=float(np.mean([f.progress_fraction for f in frames[64:]])),
                  updates=sum(f.offset_before != f.offset_after for f in frames),
                  accepts=sum(f.accept for f in frames), motor_unchanged=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    directory = parser.parse_args().output
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = Config(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    train=[Episode(motor_seed=m, scene_seed=s, scenario=c, magnitude=a)
                           for m in (41, 42) for s, a in ((4101, .15), (4102, .30), (4103, .15))
                           for c in SCENARIOS[:-1]],
                    evaluation=[Episode(motor_seed=m, scene_seed=s, scenario=c, magnitude=a, period=p)
                                for m in (41, 42, 43) for s, a, p in ((5201, .22, 48), (5202, .38, 80))
                                for c in SCENARIOS])
    (directory / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    x, benefit = collect(config, directory)
    gates = [(seed, train_gate(config, x, benefit, seed, directory)) for seed in config.gate_seeds]
    modes: list[tuple[Mode, int | None, UpdateGate | None]] = [
        ("off", None, None), ("always", None, None), ("rule", None, None),
        *(("learned", seed, gate) for seed, gate in gates)]
    for index, episode in enumerate(config.evaluation):
        baseline: Result | None = None
        for mode, gate_seed, gate in modes:
            path = directory / f"episode-{index:02d}-{mode}-{gate_seed}"
            path.mkdir()
            result = evaluate(episode, mode, gate_seed, gate, path)
            if baseline is None:
                baseline = result
            result.error_difference = result.mean_error_degrees - baseline.mean_error_degrees
            result.progress_difference = result.progress_fraction - baseline.progress_fraction
            with (directory / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(result.model_dump_json() + "\n")
        print(f"留出对照 {index + 1}/{len(config.evaluation)}", flush=True)


if __name__ == "__main__":
    main()
