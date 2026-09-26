"""陌生混合信号下接受/跳过候选的短试验，尚不训练学习门。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel
import torch

from direction_adaptation import FOUNDATION
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.adaptation import AdaptationConfig, AdaptationProposal, NovelSignalLearner
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony
from mathhackson.training.foraging.policy import ForagingPolicy

Mode = Literal["skip", "accept"]
Condition = Literal["benign", "persistent"]


class Artifact(BaseModel):
    path: str
    sha256: str


class ProbeConfig(BaseModel):
    source_commit: str
    sources: list[Artifact]
    seeds: tuple[int, ...] = (9401, 9402)
    generations: int = 3
    modes: tuple[Mode, ...] = ("skip", "accept")
    conditions: tuple[Condition, ...] = ("benign", "persistent")
    environment: ColonyConfig = ColonyConfig()
    adaptation: AdaptationConfig = AdaptationConfig()
    disturbance: DisturbanceConfig = DisturbanceConfig()


class ProposalRecord(BaseModel):
    number: int
    delta: list[float]
    gradient_norm: float
    mean_reward: float
    minimum_reward: float
    novel_response: float
    steps: int
    accepted: bool
    changed: bool


class FrameAnt(BaseModel):
    index: int
    active: bool
    observation: list[float]
    hidden: list[float]
    x: float
    y: float
    heading: float
    move: bool
    turn: float
    reward: float
    delivered: bool
    killed: bool
    exhausted: bool
    proposal: ProposalRecord | None


class Frame(BaseModel):
    tick: int
    ants: list[FrameAnt]


class Trial(BaseModel):
    condition: Condition
    mode: Mode
    seed: int
    generation: int
    ticks: int
    pickups: int
    deliveries: int
    deaths: int
    budget_exhaustions: int
    injury: float
    reward: float
    proposals: int
    writes: int
    residual_norms: list[float]


def record(proposal: AdaptationProposal, accepted: bool, changed: bool) -> ProposalRecord:
    return ProposalRecord(**asdict(proposal), accepted=accepted, changed=changed)


def run(agents: list[NovelSignalLearner], config: ProbeConfig, seed: int, condition: Condition,
        mode: Mode, generation: int, directory: Path) -> Trial:
    disturbance = (config.disturbance.model_copy(update={"injury_per_step": 0.})
                   if condition == "benign" else config.disturbance)
    env = DisturbedColony(seed, config.environment, disturbance)
    for i, agent in enumerate(agents):
        agent.restart()
        agent.random.manual_seed(seed * 128 + generation * 8 + i)
    before_decisions = sum(agent.decisions for agent in agents)
    before_writes = sum(agent.writes for agent in agents)
    total_reward = 0.
    with gzip.open(directory / f"generation-{generation:02d}.jsonl.gz", "xt", encoding="utf-8") as stream:
        while not env.done:
            observed = [env.observation(i) for i in range(len(agents))]
            active = [not ant.exhausted for ant in env.ants]
            actions = [agent.act(observed[i]).action if active[i] else DirectionAction(False, 0., 0.)
                       for i, agent in enumerate(agents)]
            events = env.step(actions)
            frames = []
            for i, agent in enumerate(agents):
                ant, event = env.ants[i], events[i]
                proposal = None
                total_reward += event.reward
                if active[i]:
                    agent.feedback(event.reward, terminal=event.exhausted or env.done)
                    if agent.ready:
                        item = agent.propose(env.observation(i))
                        accepted = mode == "accept"
                        changed = agent.resolve(item, accept=(accepted, accepted))
                        proposal = record(item, accepted, changed)
                frames.append(FrameAnt(index=i, active=active[i], observation=observed[i].vector().tolist(),
                                       hidden=agent.history[-1].detach().tolist() if agent.history else [],
                                       x=float(ant.position[0]), y=float(ant.position[1]), heading=ant.heading,
                                       move=actions[i].move, turn=actions[i].turn, reward=event.reward,
                                       delivered=event.delivered, killed=bool(env.killed[i]), exhausted=ant.exhausted,
                                       proposal=proposal))
            stream.write(Frame(tick=env.steps, ants=frames).model_dump_json() + "\n")
    for i, agent in enumerate(agents):
        if mode == "skip" and np.any(agent.parameter.fast):
            raise AssertionError("跳过组出现参数写入")
        agent.save(directory / f"generation-{generation:02d}-ant-{i:02d}.npz")
    result = Trial(condition=condition, mode=mode, seed=seed, generation=generation, ticks=env.steps,
                   pickups=sum(a.pickups for a in env.ants), deliveries=sum(a.deliveries for a in env.ants),
                   deaths=int(env.killed.sum()), budget_exhaustions=sum(a.exhausted for a in env.ants) - int(env.killed.sum()),
                   injury=float(env.injuries.sum()), reward=total_reward,
                   proposals=sum(a.decisions for a in agents) - before_decisions,
                   writes=sum(a.writes for a in agents) - before_writes,
                   residual_norms=[float(np.linalg.norm(a.parameter.fast)) for a in agents])
    with (directory.parent / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(result.model_dump_json() + "\n")
    print(result.model_dump_json(), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    training = Path("logs/colony-reward/20260926T070617-2")
    paths = [training / f"episode-0008-ant-{i:02d}.npz" for i in range(8)]
    motor_path = FOUNDATION / "seed-41/update-001200.npz"
    config = ProbeConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         sources=[Artifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in [*paths, motor_path]])
    if args.smoke:
        config = config.model_copy(update={"seeds": (9499,), "generations": 2,
                                           "environment": ColonyConfig(ants=2, horizon=20)})
    (args.output / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    motor, _ = load_motor(motor_path)
    for condition in config.conditions:
        for seed in config.seeds:
            for mode in config.modes:
                directory = args.output / f"{condition}-{seed}-{mode}"
                directory.mkdir()
                agents = [NovelSignalLearner(ForagingPolicy.load(paths[i]), motor, seed * 8 + i, config.adaptation)
                          for i in range(config.environment.ants)]
                base = [{key: value.detach().clone() for key, value in agent.policy.state_dict().items()
                         if not key.startswith(("novel_signal.", "novel_strength."))} for agent in agents]
                for i, agent in enumerate(agents):
                    agent.save(directory / f"generation-00-ant-{i:02d}.npz")
                for generation in range(1, config.generations + 1):
                    run(agents, config, seed, condition, mode, generation, directory)
                    for i, agent in enumerate(agents):
                        if any(not value.equal(agent.policy.state_dict()[key]) for key, value in base[i].items()):
                            raise AssertionError("基础导航参数被在线更新")
                        if any(not a.equal(b) for a, b in zip(motor.parameters(), agent.motor.parameters(), strict=True)):
                            raise AssertionError("动作底座被在线更新")
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in config.sources):
        raise AssertionError("源检查点发生改变")


if __name__ == "__main__":
    main()
