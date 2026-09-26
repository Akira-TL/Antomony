"""按结果前配置采集候选更新分支，不训练门控或改变主轨迹。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.policy import DirectionAction
from .adaptation import AdaptationConfig, AdaptationProposal, NovelDirectionLearner
from .candidate_value import CandidateValue, evaluate_candidate
from .colony import ColonyConfig
from .disturbance import DisturbanceConfig, DisturbedColony
from .policy import ForagingPolicy
from .trust_candidate import CandidateDiagnostics, TrustConfig, TrustDirectionLearner


class ProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...]
    conditions: tuple[Literal["benign", "persistent"], ...]
    pairs_per_world: int = Field(ge=1, le=32)
    branch_horizon: int = Field(ge=1, le=64)
    environment: ColonyConfig
    candidate_kind: Literal["gradient", "trust"] = "gradient"
    adaptation: AdaptationConfig | TrustConfig
    disturbance: DisturbanceConfig
    policy_directory: str
    motor_path: str
    initial_residual_norm: float = Field(default=0., ge=0., allow_inf_nan=False)

    @model_validator(mode="after")
    def check_candidate_configuration(self) -> ProbePlan:
        if (self.candidate_kind == "trust") != isinstance(self.adaptation, TrustConfig):
            raise ValueError("候选类型与参数配置不一致")
        if self.initial_residual_norm >= self.adaptation.maximum_residual_norm:
            raise ValueError("初始残差必须小于在线残差安全上限")
        return self


class SourceArtifact(BaseModel):
    path: str
    sha256: str


class Execution(BaseModel):
    source_commit: str
    plan_path: str
    plan_sha256: str
    smoke: bool
    plan: ProbePlan
    sources: list[SourceArtifact]


class CandidateRecord(BaseModel):
    condition: str
    seed: int
    tick: int
    focal: int
    observation: list[float]
    hidden: list[float]
    proposal: AdaptationProposal
    result: CandidateValue
    diagnostics: CandidateDiagnostics | None = None


class WorldRecord(BaseModel):
    condition: str
    seed: int
    steps: int
    pairs: int
    zero_candidates: int
    terminal_candidates: int
    deliveries: int
    deaths: int


def initialize_residual(agent: NovelDirectionLearner, *, seed: int, norm: float) -> None:
    if (not np.isfinite(norm) or norm < 0. or norm >= agent.config.maximum_residual_norm
            or agent.history or agent.awaiting_feedback or agent.rewards or agent.proposal is not None
            or agent.decisions or agent.writes or np.any(agent.parameter.fast)):
        raise ValueError("初始扰动只能用于尚未行动的新个体且必须在安全范围内")
    if norm == 0.:
        return
    random = np.random.default_rng(seed)
    vector = random.normal(size=agent.parameter.fast.shape)
    vector *= norm / np.linalg.norm(vector)
    agent.parameter.fast = vector.astype(np.float32)
    agent.assign_weights(agent.parameter.effective)


def collect(plan: ProbePlan, seed: int, condition: str, directory: Path) -> WorldRecord:
    motor, _ = load_motor(Path(plan.motor_path))
    learner = TrustDirectionLearner if plan.candidate_kind == "trust" else NovelDirectionLearner
    agents = [learner(ForagingPolicy.load(Path(plan.policy_directory) / f"episode-0008-ant-{i:02d}.npz"),
                                    motor, seed * 8 + i, plan.adaptation) for i in range(plan.environment.ants)]
    for i, agent in enumerate(agents):
        initialize_residual(agent, seed=seed * 8 + i, norm=plan.initial_residual_norm)
        if plan.initial_residual_norm:
            agent.save(directory / f"tick-0000-ant-{i:02d}.npz")
    initial = [agent.parameter.fast.copy() for agent in agents]
    disturbance = plan.disturbance if condition == "persistent" else plan.disturbance.model_copy(update={"injury_per_step": 0.})
    env = DisturbedColony(seed, plan.environment, disturbance)
    pairs = zeros = terminals = 0
    with (directory / "pairs.jsonl").open("x", encoding="utf-8") as output, gzip.open(directory / "parent.jsonl.gz", "xt") as parent:
        while not env.done:
            active = [not ant.exhausted for ant in env.ants]
            actions = [agent.act(env.observation(i)).action if active[i] else DirectionAction(False, 0., 0.)
                       for i, agent in enumerate(agents)]
            events = env.step(actions)
            eligible: list[int] = []
            for i, agent in enumerate(agents):
                if not active[i]:
                    continue
                agent.feedback(events[i].reward, terminal=env.done or events[i].exhausted)
                if agent.ready:
                    proposal = agent.propose(env.observation(i))
                    if agent.terminal:
                        terminals += 1
                    elif not np.any(proposal.delta):
                        zeros += 1
                    else:
                        eligible.append(i)
            if eligible and pairs < plan.pairs_per_world and not env.done:
                start = (env.steps // plan.adaptation.window - 1) % len(agents)
                focal = min(eligible, key=lambda i: (i - start) % len(agents))
                agent = agents[focal]
                result = evaluate_candidate(env, agents, focal, horizon=plan.branch_horizon)
                record = CandidateRecord(condition=condition, seed=seed, tick=env.steps, focal=focal,
                                         observation=env.observation(focal).vector().tolist(),
                                         hidden=agent.history[-1].detach().tolist(), proposal=agent.proposal, result=result,
                                         diagnostics=agent.diagnostics if isinstance(agent, TrustDirectionLearner) else None)
                output.write(record.model_dump_json() + "\n")
                output.flush()
                pairs += 1
            for agent in agents:
                if agent.proposal is not None:
                    agent.resolve(agent.proposal, accept=(False,))
            # 只记录主轨迹的实际动作；所有分支未来输出与它隔离。
            parent.write(ParentFrame(tick=env.steps, moves=[a.move for a in actions], turns=[a.turn for a in actions],
                                     rewards=[e.reward for e in events]).model_dump_json() + "\n")
            if env.steps % 64 == 0 or env.done:
                for i, agent in enumerate(agents):
                    agent.save(directory / f"tick-{env.steps:04d}-ant-{i:02d}.npz")
    if any(agent.writes or not np.array_equal(agent.parameter.fast, before)
           for agent, before in zip(agents, initial, strict=True)):
        raise AssertionError("分支评价污染了主轨迹参数")
    return WorldRecord(condition=condition, seed=seed, steps=env.steps, pairs=pairs, zero_candidates=zeros,
                       terminal_candidates=terminals, deliveries=sum(a.deliveries for a in env.ants), deaths=int(env.killed.sum()))


class ParentFrame(BaseModel):
    tick: int
    moves: list[bool]
    turns: list[float]
    rewards: list[float]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = ProbePlan.model_validate_json(args.plan.read_text())
    if args.smoke:
        plan = plan.model_copy(update={"seeds": (9599,), "pairs_per_world": 1, "branch_horizon": 3,
                                       "environment": plan.environment.model_copy(update={"ants": 2, "horizon": 20})})
    torch.set_num_threads(1)
    paths = [Path(plan.policy_directory) / f"episode-0008-ant-{i:02d}.npz" for i in range(plan.environment.ants)]
    sources = [SourceArtifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for path in [*paths, Path(plan.motor_path)]]
    args.output.mkdir(parents=True, exist_ok=False)
    execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          plan_path=str(args.plan), plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                          smoke=args.smoke, plan=plan, sources=sources)
    (args.output / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (args.output / "worlds.jsonl").open("x", encoding="utf-8") as output:
        for condition in plan.conditions:
            for seed in plan.seeds:
                directory = args.output / f"{condition}-{seed}"
                directory.mkdir()
                result = collect(plan, seed, condition, directory)
                output.write(result.model_dump_json() + "\n")
                output.flush()
                print(result.model_dump_json(), flush=True)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in sources):
        raise AssertionError("源模型在分支试验中改变")


if __name__ == "__main__":
    main()
