"""固定预算的群体往返开发试验；不宣称已完成自修改训练。"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import gzip
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from direction_adaptation import FOUNDATION
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.reward import DIRECTIONS, actor_critic_losses, direction_distribution, discounted_returns


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_commit: str
    signal_path: str
    signal_sha256: str
    motor_path: str
    motor_sha256: str
    signal_updates: int = Field(default=1200, ge=1)
    episodes: int = Field(default=8, ge=1)
    checkpoint_every: int = 2
    unroll: int = 32
    gamma: float = .995
    learning_rate: float = .0003
    signal_learning_rate: float = .003
    weight_decay: float = .0001
    signal_anchor: float = .25
    critic_weight: float = .5
    gradient_clip: float = 1.
    training_seed: int = 11000
    teacher_seed: int = 7201
    evaluation_seeds: tuple[int, ...] = (9201, 9202, 9203, 9204)
    environment: ColonyConfig = ColonyConfig()


class TrainingRecord(BaseModel):
    stage: str
    update: int
    loss: float
    mean_angle: float
    empty_return_angle: float
    within_15: float


class AntFrame(BaseModel):
    index: int
    active: bool
    observation: list[float]
    hidden: list[float]
    direction: list[float]
    x: float
    y: float
    heading: float
    move: bool
    turn: float
    picked_up: bool
    delivered: bool
    budget_return: bool
    exhausted: bool
    reward: float
    updates: int


class WorldFrame(BaseModel):
    tick: int
    ants: list[AntFrame]


class Trial(BaseModel):
    stage: str
    seed: int
    sampled: bool
    trails_visible: bool
    training: bool
    ticks: int
    pickups: int
    deliveries: int
    budget_returns: int
    empty_budget_returns: int
    exhausted: int
    reward: float
    updates: list[int]
    food_available: bool = True


def assert_reserved(model: ForagingPolicy) -> None:
    if any(bool(p.any()) or p.grad is not None for p in model.reserved_parameters()):
        raise AssertionError("预留接收器被基础课程改动")


def budget_pretrain(config: RunConfig, directory: Path) -> tuple[Path, Path]:
    model = ForagingPolicy.load(Path(config.signal_path)).with_budgets()
    model.set_phase("signal")
    rng = np.random.default_rng(config.teacher_seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.signal_learning_rate, weight_decay=config.weight_decay)
    initial = directory / "budget-000000.npz"
    model.save(initial, update=0, phase="signal")
    validation, target = signal_batch(np.random.default_rng(8201), 2048, budgets=True)
    for update in range(1, config.signal_updates + 1):
        inputs, expected = signal_batch(rng, 256, budgets=True)
        loss = 1. - (model(inputs)[0] * expected).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        assert_reserved(model)
        if update % 100 == 0 or update == config.signal_updates:
            model.save(directory / f"budget-{update:06d}.npz", update=update, phase="signal")
            torch.save(optimizer.state_dict(), directory / f"budget-{update:06d}-optimizer.pt")
            with torch.inference_mode():
                errors = torch.rad2deg(torch.acos((model(validation)[0] * target).sum(-1).clamp(-1., 1.)))
            empty_return = (validation[:, 72] == 0.) & (validation[:, 76] == 0.)
            record = TrainingRecord(stage="budget", update=update, loss=float(loss.detach()),
                                    mean_angle=float(errors.mean()), empty_return_angle=float(errors[empty_return].mean()),
                                    within_15=float((errors < 15.).float().mean()))
            with (directory / "signal-training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
            print(record.model_dump_json(), flush=True)
    return initial, directory / f"budget-{config.signal_updates:06d}.npz"


class Learner:
    def __init__(self, checkpoint: Path, config: RunConfig, index: int, training: bool) -> None:
        self.config = config
        self.model = ForagingPolicy.load(checkpoint)
        self.model.set_phase("reward" if training else "frozen")
        self.motor, _ = load_motor(Path(config.motor_path))
        self.motor.freeze()
        self.motor_before = [p.detach().clone() for p in self.motor.parameters()]
        self.rng = np.random.default_rng(config.teacher_seed + 100 + index)
        self.optimizer = (torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=config.learning_rate, weight_decay=config.weight_decay) if training else None)
        self.history: deque[torch.Tensor] = deque(maxlen=16)
        self.log_probabilities: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.updates = 0

    def update(self, next_observation: torch.Tensor, terminal: bool) -> None:
        if self.optimizer is None or not self.rewards:
            return
        with torch.no_grad():
            bootstrap = 0. if terminal else float(self.model(next_observation, tuple(self.history))[2])
        returns = discounted_returns(self.rewards, bootstrap, self.config.gamma)
        actor, critic = actor_critic_losses(torch.stack(self.log_probabilities), torch.stack(self.values), returns)
        inputs, target = signal_batch(self.rng, 64, budgets=True)
        anchor = 1. - (self.model(inputs)[0] * target).sum(-1).mean()
        loss = actor + self.config.critic_weight * critic + self.config.signal_anchor * anchor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.history = deque((h.detach() for h in self.history), maxlen=16)
        self.log_probabilities.clear()
        self.values.clear()
        self.rewards.clear()
        self.updates += 1
        assert_reserved(self.model)


def run_world(learners: list[Learner], config: RunConfig, seed: int, stage: str, directory: Path,
              *, sampled: bool, visible: bool = True, training: bool = False, food_available: bool = True) -> Trial:
    env = ColonyEnvironment(seed, config.environment)
    if not food_available:
        env.stock = 0
    generators = [torch.Generator().manual_seed(seed * 32 + i) for i in range(len(learners))]
    before = [[p.detach().clone() for p in agent.model.parameters()] for agent in learners]
    for agent in learners:
        if agent.rewards:
            raise AssertionError("上一世界还有未完成的个体更新")
        agent.history.clear()
    total_reward, empty_returns = 0., 0
    trace = directory / f"{stage}-{seed}-sample-{sampled}-trails-{visible}.jsonl.gz"
    with gzip.open(trace, "xt", encoding="utf-8") as stream:
        while not env.done:
            observations, chosen, active = [], [], []
            actions = []
            for i, learner in enumerate(learners):
                observation = env.observation(i)
                if not visible:
                    receptors = observation.receptors.copy()
                    receptors[:, 1:3] = 0.
                    observation = replace(observation, receptors=receptors)
                tensor = torch.from_numpy(observation.vector())
                observations.append(tensor)
                active.append(not env.ants[i].exhausted)
                if not active[-1]:
                    actions.append(DirectionAction(False, 0., 0.))
                    chosen.append(torch.tensor([1., 0.]))
                    continue
                with torch.set_grad_enabled(training):
                    direction, hidden, value = learner.model(tensor, tuple(learner.history))
                    learner.history.append(hidden)
                    if sampled:
                        distribution = direction_distribution(direction)
                        index = torch.multinomial(distribution.probs.detach(), 1, generator=generators[i]).squeeze(0)
                        chosen.append(DIRECTIONS[index])
                        if training:
                            learner.log_probabilities.append(distribution.log_prob(index))
                            learner.values.append(value)
                    else:
                        chosen.append(direction.detach())
                actions.append(learner.motor.decide(chosen[-1].numpy()))
            outcomes = env.step(actions)
            frames = []
            for i, (learner, event, action) in enumerate(zip(learners, outcomes, actions, strict=True)):
                ant = env.ants[i]
                total_reward += event.reward
                empty_returns += int(event.budget_return and not event.delivered)
                if training and active[i]:
                    learner.rewards.append(event.reward)
                    terminal = env.done or event.exhausted
                    if len(learner.rewards) >= config.unroll or terminal:
                        learner.update(torch.from_numpy(env.observation(i).vector()), terminal)
                frames.append(AntFrame(index=i, active=active[i], observation=observations[i].tolist(),
                                       hidden=learner.history[-1].detach().tolist() if learner.history else [],
                                       direction=chosen[i].detach().tolist(), x=float(ant.position[0]), y=float(ant.position[1]),
                                       heading=ant.heading, move=action.move, turn=action.turn, picked_up=event.picked_up,
                                       delivered=event.delivered, budget_return=event.budget_return, exhausted=ant.exhausted,
                                       reward=event.reward, updates=learner.updates))
            stream.write(WorldFrame(tick=env.steps, ants=frames).model_dump_json() + "\n")
    for i, learner in enumerate(learners):
        assert_reserved(learner.model)
        if not all(a.equal(b) for a, b in zip(learner.motor_before, learner.motor.parameters(), strict=True)):
            raise AssertionError("冻结动作底座被改动")
        if not training and not all(a.equal(b) for a, b in zip(before[i], learner.model.parameters(), strict=True)):
            raise AssertionError("评估期间策略改变")
    result = Trial(stage=stage, seed=seed, sampled=sampled, trails_visible=visible, training=training, ticks=env.steps,
                   pickups=sum(a.pickups for a in env.ants), deliveries=sum(a.deliveries for a in env.ants),
                   budget_returns=sum(a.budget_returns for a in env.ants), empty_budget_returns=empty_returns,
                   exhausted=sum(a.exhausted for a in env.ants), reward=total_reward, updates=[a.updates for a in learners],
                   food_available=food_available)
    with (directory / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(result.model_dump_json() + "\n")
    print(result.model_dump_json(), flush=True)
    return result


def save_learners(learners: list[Learner], directory: Path, episode: int) -> list[Path]:
    paths = []
    for i, learner in enumerate(learners):
        path = directory / f"episode-{episode:04d}-ant-{i:02d}.npz"
        learner.model.save(path, update=learner.updates, phase="reward")
        if learner.optimizer is not None:
            torch.save(learner.optimizer.state_dict(), path.with_suffix(".optimizer.pt"))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    signal_path = Path("logs/local-signal/20260926T064239-2/signal-001200.npz")
    motor_path = FOUNDATION / "seed-41/update-001200.npz"
    config = RunConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       signal_path=str(signal_path), signal_sha256=hashlib.sha256(signal_path.read_bytes()).hexdigest(),
                       motor_path=str(motor_path), motor_sha256=hashlib.sha256(motor_path.read_bytes()).hexdigest())
    if args.smoke:
        config = config.model_copy(update={"signal_updates": 2, "episodes": 1, "evaluation_seeds": (9299,),
                                           "environment": ColonyConfig(ants=2, horizon=8)})
    (args.output / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    initial, budget = budget_pretrain(config, args.output)
    learners = [Learner(budget, config, i, True) for i in range(config.environment.ants)]
    if len({p.data_ptr() for learner in learners for p in learner.model.parameters()}) != sum(len(list(a.model.parameters())) for a in learners):
        raise AssertionError("个体参数共享存储")
    save_learners(learners, args.output, 0)
    for episode in range(1, config.episodes + 1):
        run_world(learners, config, config.training_seed + episode, "training", args.output, sampled=True, training=True)
        if episode % config.checkpoint_every == 0 or episode == config.episodes:
            last = save_learners(learners, args.output, episode)
    stages = (("initial", [initial] * config.environment.ants), ("budget", [budget] * config.environment.ants), ("reward", last))
    for stage, checkpoints in stages:
        for seed in config.evaluation_seeds:
            for sampled in (False, True):
                agents = [Learner(path, config, i, False) for i, path in enumerate(checkpoints)]
                run_world(agents, config, seed, stage, args.output, sampled=sampled)
    for seed in config.evaluation_seeds:
        agents = [Learner(path, config, i, False) for i, path in enumerate(last)]
        run_world(agents, config, seed, "reward", args.output, sampled=True, visible=False)
    if (hashlib.sha256(signal_path.read_bytes()).hexdigest() != config.signal_sha256
            or hashlib.sha256(motor_path.read_bytes()).hexdigest() != config.motor_sha256):
        raise AssertionError("源检查点文件发生变化")


if __name__ == "__main__":
    main()
