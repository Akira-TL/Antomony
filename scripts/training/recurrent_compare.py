"""从单蚁检查点执行只读、确定性的同种子写入方式对照。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from mathhackson.training.environment import SingleAntEnvironment
from mathhackson.training.recurrent import RecurrentPolicy, WriteMode
from mathhackson.training.recurrent_checkpoint import PARAMETER_NAMES, read_recurrent_parameters

PARAMETERS = PARAMETER_NAMES
TASKS = ("normal", "shift", "sensor")
MODES: tuple[WriteMode, ...] = ("off", "learned", "always")


@dataclass(frozen=True)
class Trial:
    seed: int
    task: str
    mode: WriteMode
    perturbation: float
    reached: bool
    steps: int
    reward: float
    writes: int


def read_parameters(checkpoint: Path) -> tuple[np.ndarray, ...]:
    return read_recurrent_parameters(checkpoint)


def run_trial(parameters: tuple[np.ndarray, ...], seed: int, task: str, mode: WriteMode) -> Trial:
    policy = RecurrentPolicy(0)
    with torch.no_grad():
        for name, values in zip(PARAMETERS, parameters, strict=True):
            parameter = getattr(policy, name)
            if parameter.shape != values.shape or not np.isfinite(values).all():
                raise ValueError(f"检查点参数 {name} 的形状或数值无效")
            parameter.copy_(torch.from_numpy(values))
    policy.phase = "autonomous"
    policy.write_mode = mode
    environment = SingleAntEnvironment(seed)
    sensor_magnitude = float(np.random.default_rng(seed + 73409).uniform(.55, .75)) if task == "sensor" else 0.
    perturbation = 0.
    total_reward = 0.
    while not environment.done:
        if environment.steps == 8 and task == "shift":
            perturbation = -.25 if seed % 2 else .25
            environment.turn_bias = perturbation
        if environment.steps == 8 and task == "sensor":
            perturbation = -sensor_magnitude if seed % 2 else sensor_magnitude
            environment.sensor_bias = perturbation
        action = policy.decide(environment.observation())
        total_reward += environment.step(action.move, action.turn)
        policy.observe_result(environment.observation(), terminal=environment.done)
    return Trial(seed, task, mode, perturbation, environment.reached, environment.steps,
                 total_reward, policy.self_updates)


def compare(checkpoint: Path, first_seed: int, seeds: int) -> dict[str, object]:
    parameters = read_parameters(checkpoint)
    trials = [run_trial(parameters, seed, task, mode)
              for seed in range(first_seed, first_seed + seeds)
              for task in TASKS for mode in MODES]
    indexed = {(trial.seed, trial.task, trial.mode): trial for trial in trials}
    pairs = []
    for seed in range(first_seed, first_seed + seeds):
        for task in TASKS:
            baseline = indexed[seed, task, "off"]
            for mode in MODES[1:]:
                current = indexed[seed, task, mode]
                pairs.append({"seed": seed, "task": task, "mode": mode,
                              "perturbation": current.perturbation,
                              "reached": current.reached, "baseline_reached": baseline.reached,
                              "steps": current.steps, "baseline_steps": baseline.steps,
                              "step_difference": current.steps - baseline.steps,
                              "reward_difference": current.reward - baseline.reward,
                              "writes": current.writes})
    summary = []
    for task in TASKS:
        baseline = [trial for trial in trials if trial.task == task and trial.mode == "off"]
        for mode in MODES:
            selected = [trial for trial in trials if trial.task == task and trial.mode == mode]
            summary.append({"task": task, "mode": mode,
                            "reached": sum(trial.reached for trial in selected),
                            "total": len(selected),
                            "mean_steps_including_failures": sum(trial.steps for trial in selected) / seeds,
                            "mean_reward": sum(trial.reward for trial in selected) / seeds,
                            "mean_writes": sum(trial.writes for trial in selected) / seeds,
                            "reach_difference": sum(trial.reached for trial in selected) -
                                                sum(trial.reached for trial in baseline),
                            "mean_step_difference": (sum(trial.steps for trial in selected) -
                                                     sum(trial.steps for trial in baseline)) / seeds})
    return {"checkpoint": str(checkpoint), "first_seed": first_seed, "seeds": seeds,
            "summary": summary, "pairs": pairs, "trials": [asdict(trial) for trial in trials],
            "interpretation": "工程诊断；不能作为正式学习优势证据"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds 必须大于零")
    print(json.dumps(compare(args.checkpoint, args.first_seed, args.seeds),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
