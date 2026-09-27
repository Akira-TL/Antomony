"""固定预算的离线基础课程；工程计时与正式留出严格分开。"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from mathhackson.training.direction.checkpoint import load_motor
from ..feedback_cues import CONDITIONS, Condition, CueConfig, CueEpisode, cue_episode
from ..memory import MemoryPolicy
from ..plastic_direction import DirectionMode, PlasticDirection
from .checkpoint import save_checkpoint
from .rollout import CourseTrace, Mode, matched_schedule, outer_loss, paired_baseline, rollout


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Initialization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int
    foundation: Source
    train_seed: int
    evaluation_seed: int


class Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["feedback-plasticity-course-v1"] = "feedback-plasticity-course-v1"
    motor: Source
    initializations: tuple[Initialization, ...]
    course: CueConfig = CueConfig()
    updates: int = Field(default=150, ge=1, le=300)
    evaluation_batches: int = Field(default=4, ge=1, le=8)
    learning_rate: float = Field(default=.003, gt=0, le=.01)
    weight_decay: float = Field(default=.0001, ge=0, le=.01)
    gradient_norm: float = Field(default=1., gt=0, le=10.)
    gamma: float = Field(default=.97, ge=0, le=1.)
    baseline_decay: float = Field(default=.9, ge=0, lt=1.)
    write_cost: float = Field(default=.002, ge=0, le=.1)
    max_step: float = Field(default=.25, gt=0, le=2.)
    max_fast: float = Field(default=2., gt=0, le=4.)
    direction_mode: DirectionMode = "unit"
    evaluate_initial: bool = False
    paired_courses: bool = False
    baseline_mode: Literal["history", "paired"] = "history"
    checkpoint_interval: Literal[25] = 25
    time_limit_seconds: int = Field(default=900, ge=30, le=900)

    @model_validator(mode="after")
    def distinct_partitions(self) -> Protocol:
        if len(self.initializations) != 2 or len({item.seed for item in self.initializations}) != 2:
            raise ValueError("基础可行性课程固定两个不同的外层初始化")
        partitions: list[set[int]] = []
        for item in self.initializations:
            partitions.extend((set(range(item.train_seed + 1, item.train_seed + self.updates + 1)),
                set(range(item.evaluation_seed, item.evaluation_seed + self.evaluation_batches))))
        seen = {908101, 908102}
        for partition in partitions:
            if seen & partition:
                raise ValueError("训练、留出与工程计时种子不可重叠")
            seen.update(partition)
        if self.max_step > self.max_fast:
            raise ValueError("单次写入范数上限不能超过累计上限")
        if self.paired_courses and self.course.batch % 2:
            raise ValueError("成对训练课程需要偶数批量")
        if self.baseline_mode == "paired" and not self.paired_courses:
            raise ValueError("同对另一轨迹基线需要成对训练课程")
        return self


class TrainingRow(BaseModel):
    update: int
    loss: float
    mean_reward: float
    accepted_fraction: float
    gradient_norm: float
    elapsed_seconds: float


class EvaluationRow(BaseModel):
    initialization: int
    condition: Condition
    batch_seed: int
    mode: Mode
    mean_after_feedback: list[float]
    mean_last_quarter: list[float]
    acceptances: list[int]
    nonzero_writes: list[int]
    total_variation_from_initial: list[float]


class Timing(BaseModel):
    purpose: str = "仅工程计时和有限梯度检查，不计算正式留出效果"
    seed: int = 908101
    updates: int = 2
    course: CueConfig
    elapsed_seconds: float
    seconds_per_update: float
    finite_gradients: bool


class Completion(BaseModel):
    complete: bool
    reason: str
    elapsed_seconds: float
    initializations_completed: int


def verified(source: Source) -> Path:
    path = Path(source.path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
        raise ValueError(f"源模型散列不匹配：{path}")
    return path


def base_directions(base: MemoryPolicy, episode: CueEpisode) -> torch.Tensor:
    history: tuple[torch.Tensor, ...] = ()
    directions = []
    with torch.no_grad():
        for observation in episode.observations:
            direction, hidden, _ = base(observation, history)
            history = (*history, hidden)[-16:]
            directions.append(direction)
    return torch.stack(directions).detach()


def training_episode(seed: int, plan: Protocol) -> CueEpisode:
    if not plan.paired_courses:
        return cue_episode(seed, plan.course)
    half = cue_episode(seed, plan.course.model_copy(update={"batch": plan.course.batch // 2}))
    return CueEpisode(config=plan.course,
        conditions=tuple(condition for condition in half.conditions for _ in range(2)),
        observations=half.observations.repeat_interleave(2, dim=1),
        targets=half.targets.repeat_interleave(2, dim=1),
        clean_targets=half.clean_targets.repeat_interleave(2, dim=1),
        cue_ids=half.cue_ids.repeat_interleave(2, dim=1),
        reward_flips=half.reward_flips.repeat_interleave(2, dim=1))


def save_trace(path: Path, episode: CueEpisode, base: torch.Tensor, trace: CourseTrace, *, structure_update: int) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, structure_update=structure_update, observations=episode.observations.numpy(),
            base=base.numpy(), conditions=np.asarray(episode.conditions),
            targets=episode.targets.numpy(), cue_ids=episode.cue_ids.numpy(),
            reward_flips=episode.reward_flips.numpy(), rewards=trace.rewards.numpy(),
            probabilities=trace.probabilities.numpy(), directions=trace.directions.numpy(),
            accepted=trace.accepted.numpy(), modulation=trace.modulation.numpy(),
            gate_probability=trace.gate_probability.numpy(), write_norm=trace.write_norm.numpy(),
            fast=trace.fast.numpy(), eligibility=trace.eligibility.numpy(), hidden=trace.hidden.numpy())


def benchmark(output: Path) -> None:
    config = CueConfig(steps=96, batch=16)
    motor, _ = load_motor(Path("models/interactive/motor/update-001200.npz"))
    base = MemoryPolicy.load(Path("models/interactive/memory/episode-0000-ant-00.npz"))
    model = PlasticDirection(motor, seed=908100, max_step=.25, max_fast=2.)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
    baseline = torch.zeros(config.steps)
    start = time.monotonic()
    for update in range(2):
        episode = cue_episode(908101 + update, config)
        trace = rollout(model, episode, base_directions(base, episode), seed=908201 + update, training=True)
        loss, _ = outer_loss(trace, baseline)
        optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("工程计时产生非有限梯度")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    elapsed = time.monotonic() - start
    record = Timing(course=config, elapsed_seconds=elapsed, seconds_per_update=elapsed / 2, finite_gradients=True)
    (output / "timing.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    print(record.model_dump_json(), flush=True)


def train(model: PlasticDirection, base: MemoryPolicy, initialization: Initialization,
          plan: Protocol, output: Path, deadline: float) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=plan.learning_rate, weight_decay=plan.weight_decay)
    baseline = torch.zeros(plan.course.steps)
    start = time.monotonic()
    save_checkpoint(output / "step-0000.npz", model, update=0, seed=initialization.seed)
    for update in range(1, plan.updates + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("达到预定总计算预算，保留所有中间记录；未完成不作通过判断")
        episode = training_episode(initialization.train_seed + update, plan)
        directions = base_directions(base, episode)
        trace = rollout(model, episode, directions,
            seed=initialization.train_seed + update + 10_000_000, training=True)
        control = paired_baseline(trace, gamma=plan.gamma, write_cost=plan.write_cost) if plan.baseline_mode == "paired" else baseline
        loss, credit = outer_loss(trace, control, gamma=plan.gamma, write_cost=plan.write_cost)
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan.gradient_norm, error_if_nonfinite=True)
        optimizer.step()
        if plan.baseline_mode == "history":
            baseline = plan.baseline_decay * baseline + (1. - plan.baseline_decay) * credit
        row = TrainingRow(update=update, loss=float(loss.detach()), mean_reward=float(trace.rewards.mean()),
            accepted_fraction=float(trace.accepted[3::4].float().mean()), gradient_norm=float(norm),
            elapsed_seconds=time.monotonic() - start)
        with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(row.model_dump_json() + "\n")
        if update % plan.checkpoint_interval == 0 or update == plan.updates:
            save_checkpoint(output / f"step-{update:04d}.npz", model, update=update, seed=initialization.seed)
            save_trace(output / f"training-{update:04d}.npz", episode, directions, trace, structure_update=update - 1)
            print(f"seed={initialization.seed} {row.model_dump_json()}", flush=True)


def evaluate(model: PlasticDirection, base: MemoryPolicy, initialization: Initialization,
             plan: Protocol, output: Path, deadline: float, *, structure_update: int | None = None) -> None:
    structure_update = plan.updates if structure_update is None else structure_update
    if type(structure_update) is not int or structure_update < 0:
        raise ValueError("评价结构版本必须为非负整数")
    model.eval().requires_grad_(False)
    for condition in CONDITIONS:
        for batch in range(plan.evaluation_batches):
            seed = initialization.evaluation_seed + batch
            episode = cue_episode(seed, plan.course, condition=condition)
            directions = base_directions(base, episode)
            learned: CourseTrace | None = None
            for mode in ("learned", "off", "always", "matched"):
                if time.monotonic() >= deadline:
                    raise TimeoutError("评价未在固定预算内完成，不作完整通过判断")
                schedule = matched_schedule(learned.accepted, seed + 20_000_000) if mode == "matched" and learned else None
                with torch.no_grad():
                    trace = rollout(model, episode, directions, seed=seed + 10_000_000,
                        mode=mode, schedule=schedule)
                if mode == "learned":
                    learned = trace
                name = f"{condition}-{seed}-{mode}"
                save_trace(output / f"{name}.npz", episode, directions, trace, structure_update=structure_update)
                row = EvaluationRow(initialization=initialization.seed, condition=condition,
                    batch_seed=seed, mode=mode, mean_after_feedback=trace.rewards[4:].mean(dim=0).tolist(),
                    mean_last_quarter=trace.rewards[-plan.course.steps // 4:].mean(dim=0).tolist(),
                    acceptances=trace.accepted.sum(dim=0).tolist(),
                    nonzero_writes=(trace.write_norm > 1e-9).sum(dim=0).tolist(),
                    total_variation_from_initial=(.5 * (trace.probabilities - rollout_initial_probabilities(directions)).abs().sum(dim=-1)).mean(dim=0).tolist())
                with (output / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(row.model_dump_json() + "\n")


def rollout_initial_probabilities(directions: torch.Tensor) -> torch.Tensor:
    from ..reward import direction_distribution
    return direction_distribution(directions).probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--freeze-commit")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    arguments.output.mkdir(parents=True, exist_ok=False)
    if arguments.benchmark:
        if arguments.plan or arguments.freeze_commit:
            parser.error("工程计时不可读取正式分区")
        benchmark(arguments.output)
        return
    if arguments.plan is None or not arguments.freeze_commit:
        parser.error("正式实施必须提供冻结方案及其Git提交")
    stored = subprocess.check_output(["git", "show", f"{arguments.freeze_commit}:{arguments.plan.as_posix()}"])
    if stored != arguments.plan.read_bytes():
        raise ValueError("正式方案不同于冻结提交")
    plan = Protocol.model_validate_json(stored)
    (arguments.output / "protocol.json").write_bytes(stored)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (arguments.output / "commits.txt").write_text(f"code={commit}\nfreeze={arguments.freeze_commit}\n", encoding="utf-8")
    motor, _ = load_motor(verified(plan.motor))
    start, completed = time.monotonic(), 0
    try:
        for initialization in plan.initializations:
            directory = arguments.output / f"seed-{initialization.seed}"
            directory.mkdir()
            base = MemoryPolicy.load(verified(initialization.foundation))
            original = {key: value.clone() for key, value in base.state_dict().items()}
            model = PlasticDirection(motor, seed=initialization.seed, max_step=plan.max_step,
                max_fast=plan.max_fast, direction_mode=plan.direction_mode)
            if plan.evaluate_initial:
                initial = copy.deepcopy(model)
                initial_directory = arguments.output / "untrained" / f"seed-{initialization.seed}"
                initial_directory.mkdir(parents=True)
                save_checkpoint(initial_directory / "step-0000.npz", initial, update=0, seed=initialization.seed)
                evaluate(initial, base, initialization, plan, initial_directory,
                    start + plan.time_limit_seconds, structure_update=0)
            train(model, base, initialization, plan, directory, start + plan.time_limit_seconds)
            evaluate(model, base, initialization, plan, directory, start + plan.time_limit_seconds)
            if not all(torch.equal(value, original[key]) for key, value in base.state_dict().items()):
                raise AssertionError("冻结基础网络被改变")
            if not all(torch.equal(value, motor.state_dict()[key]) for key, value in model.motor.state_dict().items()):
                raise AssertionError("冻结动作网络被改变")
            completed += 1
    except TimeoutError as error:
        result = Completion(complete=False, reason=str(error), elapsed_seconds=time.monotonic() - start,
            initializations_completed=completed)
    else:
        result = Completion(complete=True, reason="固定终点训练与全部配对评价已执行；不等于效果达标",
            elapsed_seconds=time.monotonic() - start, initializations_completed=completed)
    (arguments.output / "completion.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
