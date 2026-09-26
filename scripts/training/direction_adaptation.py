"""执行运行前固定的局部方向反馈开发探针，保留负差值与逐步参数。"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from mathhackson.training.direction.adaptation import DirectionCorrection, local_direction_loss
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.environment import DirectionEnvironment, STEP_DISTANCE, wrap_angle

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = ROOT / "logs/direction-motor/20260926T053106-2"
MODELS = (41, 42, 43)
Mode = Literal["off", "sgd", "mulo"]
Scenario = Literal["normal", "positive", "negative", "transient", "periodic"]
SCENARIOS: tuple[Scenario, ...] = ("normal", "positive", "negative", "transient", "periodic")
MODES: tuple[Mode, ...] = ("off", "sgd", "mulo")


class Frame(BaseModel):
    tick: int
    desired: float
    heading: float
    x: float
    y: float
    move: bool
    turn: float
    error_degrees: float
    progress: float
    offset_before: float
    offset_after: float


class Trial(BaseModel):
    model_seed: int
    scenario: Scenario
    mode: Mode
    mean_error_degrees: float
    steady_mean_error_degrees: float
    progress_fraction: float
    updates: int
    motor_unchanged: bool
    frames: list[Frame]


class Pair(BaseModel):
    model_seed: int
    scenario: Scenario
    mode: Mode
    error_difference_degrees: float
    steady_error_difference_degrees: float
    progress_difference: float
    updates: int


class ProbeConfig(BaseModel):
    purpose: str = "在线反馈与外部更新器开发诊断，不是正式适应优势实验"
    source_commit: str
    models: tuple[int, ...] = MODELS
    scenarios: tuple[Scenario, ...] = SCENARIOS
    modes: tuple[Mode, ...] = MODES
    scene_seed: int = 3401
    horizon: int = 320
    command_interval: int = 64
    change_start: int = 96
    bias_magnitude: float = .25
    sgd_rate: float = .05
    mulo_scale: float = 1.


def perturbation(scenario: Scenario, tick: int) -> float:
    if tick < 96 or scenario == "normal":
        return 0.
    if scenario == "positive":
        return .25
    if scenario == "negative":
        return -.25
    if scenario == "transient":
        return .25 if tick < 108 else 0.
    return .25 * math.sin(2. * math.pi * (tick - 96) / 64.)


def make_optimizer(model: DirectionCorrection, mode: Mode) -> torch.optim.Optimizer | None:
    if mode == "off":
        return None
    if mode == "sgd":
        return torch.optim.SGD(model.plastic_parameters(), lr=.05)
    weights = ROOT / "logs/external-models/mulo"
    if hashlib.sha256((weights / "model.safetensors").read_bytes()).hexdigest() != (
        "ad7c499d1bd389c768f20019a6e1a423c51c7d1db64774d038f4fe56bf21ccb3"
    ):
        raise ValueError("外部权重哈希不符")
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "logs/external-models/pylo"))
    import mup
    from pylo.optim import MuLO_naive
    mup.set_base_shapes(model, model, rescale_params=False)
    optimizer = MuLO_naive(model.plastic_parameters(), lr=1., hf_key=str(weights))
    optimizer.network.requires_grad_(False)
    return optimizer


def run_trial(seed: int, scenario: Scenario, mode: Mode) -> Trial:
    motor, _ = load_motor(FOUNDATION / f"seed-{seed}/update-001200.npz")
    original = [p.detach().clone() for p in motor.parameters()]
    model = DirectionCorrection(motor)
    optimizer = make_optimizer(model, mode)
    rng = np.random.default_rng(3401)
    commands = rng.uniform(-math.pi, math.pi, 5)
    env = DirectionEnvironment(heading=0., desired=float(commands[0]))
    frames: list[Frame] = []
    for tick in range(320):
        env.desired = float(commands[tick // 64])
        before = float(model.offset.detach()[0])
        action = model.command(env.observation())
        turn = float(action.turn.detach())
        progress = env.step(action.move, turn, execution_bias=perturbation(scenario, tick))
        error = wrap_angle(env.desired - env.heading)
        if optimizer is not None:
            optimizer.zero_grad()
            local_direction_loss(action.turn, error).backward()
            optimizer.step()
            model.project()
        after = float(model.offset.detach()[0])
        frames.append(Frame(tick=tick, desired=env.desired, heading=env.heading, x=env.x, y=env.y,
                            move=action.move, turn=turn, error_degrees=abs(math.degrees(error)),
                            progress=progress, offset_before=before, offset_after=after))
    after_change = frames[96:]
    steady = [f for f in after_change if f.tick % 64 >= 32]
    unchanged = all(a.equal(b) for a, b in zip(original, motor.parameters(), strict=True))
    if not unchanged:
        raise AssertionError("冻结基础模型发生变化")
    return Trial(model_seed=seed, scenario=scenario, mode=mode,
                 mean_error_degrees=float(np.mean([f.error_degrees for f in after_change])),
                 steady_mean_error_degrees=float(np.mean([f.error_degrees for f in steady])),
                 progress_fraction=float(np.mean([f.progress / STEP_DISTANCE for f in after_change])),
                 updates=sum(f.offset_after != f.offset_before for f in frames),
                 motor_unchanged=unchanged, frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    directory = parser.parse_args().output
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = ProbeConfig(source_commit=commit)
    (directory / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    for seed in MODELS:
        for scenario in SCENARIOS:
            baseline: Trial | None = None
            for mode in MODES:
                trial = run_trial(seed, scenario, mode)
                (directory / f"{seed}-{scenario}-{mode}.json").write_text(
                    trial.model_dump_json(), encoding="utf-8")
                if baseline is None:
                    baseline = trial
                pair = Pair(model_seed=seed, scenario=scenario, mode=mode,
                            error_difference_degrees=trial.mean_error_degrees - baseline.mean_error_degrees,
                            steady_error_difference_degrees=trial.steady_mean_error_degrees - baseline.steady_mean_error_degrees,
                            progress_difference=trial.progress_fraction - baseline.progress_fraction,
                            updates=trial.updates)
                with (directory / "pairs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(pair.model_dump_json() + "\n")
                print(pair.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
