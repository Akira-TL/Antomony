"""五组隔离世界的连续部署；不调用未来分支或离线标签。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony
from mathhackson.training.foraging.revival import RevivingColony
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from mathhackson.training.foraging.trust_candidate import TrustConfig
from mathhackson.training.foraging.update_decision import UpdateDecision
from .actors import NeuralForager
from .online_actor import OnlineForager
from .run import SourceRecord

Arm = Literal["learned", "skip", "always", "mlp", "rules"]
ARMS: tuple[Arm, ...] = ("learned", "skip", "always", "mlp", "rules")
Actor = OnlineForager | NeuralForager | LocalRuleController


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["reference", "slow", "periodic", "moving-danger"]
    disturbance: DisturbanceConfig


class ContinuousPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = (18101, 18102, 18103, 18104)
    conditions: tuple[Condition, ...]
    environment: ColonyConfig = ColonyConfig(horizon=768, stock=48, trail_profile="bounded-local-v2", nest_signal_strength=1.)
    adaptation: TrustConfig = TrustConfig(feedback_mode="observed-window")
    checkpoint_every: int = Field(default=128, ge=1)
    policy_directory: str = "logs/memory-check/20260926T112254-2"
    gate_directory: str = "logs/model-artifacts/memory-acceptance"
    mlp_directory: str = "logs/matched-foundation/20260926T121232-2"
    motor: str = "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz"
    respawn: bool = False

    @model_validator(mode="after")
    def validate_plan(self) -> ContinuousPlan:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("世界种子须非空且唯一")
        if not self.conditions or len({c.name for c in self.conditions}) != len(self.conditions):
            raise ValueError("条件须非空且唯一")
        if self.adaptation.feedback_mode != "observed-window":
            raise ValueError("当前模型只能使用已发生反馈")
        return self

    def policy_path(self, index: int) -> Path:
        return Path(self.policy_directory) / f"episode-0000-ant-{index % 8:02d}.npz"

    def gate_path(self, index: int) -> Path:
        return Path(self.gate_directory) / f"ant-{index % 8:02d}" / "step-0200.npz"

    def mlp_path(self, index: int) -> Path:
        return Path(self.mlp_directory) / f"seed-{81 + index % 8}-signal-002400.npz"


class AntFrame(BaseModel):
    observation: list[float]
    active: bool
    move: bool
    turn: float
    position: list[float]
    heading: float
    carrying: bool
    exploration_left: int
    reserve_left: int
    picked_up: bool
    delivered: bool
    budget_return: bool
    exhausted: bool
    killed: bool
    injury: float
    reward: float
    writes: int
    pending: bool = False
    respawned: bool = False
    cumulative_deaths: int = 0
    cumulative_terminations: int = 0
    revivals: int = 0


class Frame(BaseModel):
    tick: int
    source_position: list[float]
    source_active: bool
    ants: list[AntFrame]
    food_stock: int | None = None


class WorldResult(BaseModel):
    seed: int
    condition: str
    arm: Arm
    steps: int
    active_individual_steps: int
    deliveries: int
    pickups: int
    budget_returns: int
    deaths: int
    exhausted: int
    injury: float
    reward: float
    decisions: int
    eligible: int
    accepted: int
    writes: list[int]
    snapshots: list[int]
    elapsed_seconds: float
    revivals: int = 0


class Execution(BaseModel):
    source_commit: str
    started_at: str
    completed_at: str | None = None
    smoke: bool
    plan: ContinuousPlan
    protocol_sha256: str
    sources: list[SourceRecord]


def make_actors(plan: ContinuousPlan, seed: int, arm: Arm) -> list[Actor]:
    if arm == "rules":
        return [LocalRuleController(seed * 32 + i) for i in range(plan.environment.ants)]
    motor, _ = load_motor(Path(plan.motor))
    if arm == "mlp":
        policies = [FeedforwardPolicy.load(plan.mlp_path(i)) for i in range(plan.environment.ants)]
        if any(policy.hidden_width != 17 for policy in policies):
            raise ValueError("普通对照须使用固定17宽模型")
        return [NeuralForager(policy, motor, seed * 32 + i) for i, policy in enumerate(policies)]
    if arm not in ("learned", "skip", "always"):
        raise ValueError("未知比较组")
    actors = []
    for i in range(plan.environment.ants):
        policy = MemoryPolicy.load(plan.policy_path(i))
        decision = UpdateDecision.load(plan.gate_path(i))
        if decision.updates != 200:
            raise ValueError("接受模型必须使用固定200步终点")
        if any(bool(p.any()) for p in policy.memory_parameters()):
            raise ValueError("当前比较要求保留基础功能的零记忆初始化")
        actors.append(OnlineForager(policy, motor, decision, seed * 32 + i, arm, plan.adaptation))
    return actors


def save_actors(actors: list[Actor], directory: Path, tick: int, *, include_memory: bool = False) -> None:
    for i, actor in enumerate(actors):
        path = directory / f"tick-{tick:04d}-ant-{i:02d}.npz"
        if isinstance(actor, OnlineForager):
            actor.save(path)
            if tick == 0:
                actor.controller.model.save(directory / f"decision-ant-{i:02d}.npz")
        elif isinstance(actor, NeuralForager):
            actor.model.save(path, update=0, phase="frozen")
        if include_memory and isinstance(actor, (OnlineForager, NeuralForager)):
            learner = actor.agent if isinstance(actor, OnlineForager) else actor
            with path.with_suffix(".memory.npz").open("xb") as stream:
                np.savez(stream, history=np.asarray([h.detach().numpy() for h in learner.history],
                    dtype=np.float32).reshape(-1, 8), random_state=learner.random.get_state().numpy())


def run_world(plan: ContinuousPlan, seed: int, condition: Condition, arm: Arm, directory: Path) -> WorldResult:
    actors = make_actors(plan, seed, arm)
    env = (RevivingColony if plan.respawn else DisturbedColony)(seed, plan.environment, condition.disturbance)
    directory.mkdir(parents=True, exist_ok=False)
    frozen = [[p.detach().clone() for p in [*a.model.parameters(), *a.motor.parameters()]]
              if isinstance(a, NeuralForager) else [] for a in actors]
    save_actors(actors, directory, 0, include_memory=plan.respawn)
    snapshots = [0]
    decisions = eligible = accepted = active_steps = 0
    reward = 0.
    started = time.perf_counter()
    with gzip.open(directory / "trajectory.jsonl.gz", "xt") as trace, (directory / "updates.jsonl").open("x") as updates:
        while not env.done:
            revived = env.release_waiting() if isinstance(env, RevivingColony) else []
            for i in revived:
                if isinstance(actors[i], OnlineForager):
                    actors[i].revive()
            active = [not ant.exhausted for ant in env.ants]
            observations = [env.observation(i) for i in range(len(actors))]
            source_position = env.source_position.tolist()
            source_active = condition.disturbance.active_at(env.steps)
            actions = []
            for actor, observation, alive in zip(actors, observations, active, strict=True):
                if not alive:
                    action = DirectionAction(False, 0., 0.)
                elif isinstance(actor, NeuralForager):
                    action = actor.act(observation, sampled=True)
                else:
                    action = actor.act(observation)
                actions.append(action)
            events = env.step(actions)
            for i, actor in enumerate(actors):
                if isinstance(actor, OnlineForager) and active[i]:
                    record = actor.feedback(events[i].reward, env.observation(i), terminal=env.done or env.ants[i].exhausted,
                                            tick=env.steps, individual=i,
                                            continuing_after_death=plan.respawn and env.ants[i].exhausted and not env.done)
                    if record is not None:
                        updates.write(record.model_dump_json() + "\n")
                        decisions += 1
                        eligible += int(record.eligible)
                        accepted += int(record.accepted)
            active_steps += sum(active)
            reward += sum(event.reward for event in events)
            trace.write(Frame(tick=env.steps, source_position=source_position, source_active=source_active,
                food_stock=env.stock if plan.respawn else None,
                ants=[AntFrame(observation=observations[i].vector().tolist(), active=active[i], move=actions[i].move,
                    turn=actions[i].turn, position=ant.position.tolist(), heading=ant.heading, carrying=ant.carrying,
                    exploration_left=ant.exploration_left, reserve_left=ant.reserve_left, picked_up=events[i].picked_up,
                    delivered=events[i].delivered, budget_return=events[i].budget_return, exhausted=ant.exhausted,
                    killed=bool(env.killed[i]), injury=float(env.injuries[i]), reward=events[i].reward,
                    writes=actors[i].agent.writes if isinstance(actors[i], OnlineForager) else 0,
                    pending=bool(env.pending[i]) if isinstance(env, RevivingColony) else False,
                    respawned=i in revived,
                    cumulative_deaths=int(env.deaths[i]) if isinstance(env, RevivingColony) else 0,
                    cumulative_terminations=int(env.terminations[i]) if isinstance(env, RevivingColony) else 0,
                    revivals=int(env.revivals[i]) if isinstance(env, RevivingColony) else 0)
                    for i, ant in enumerate(env.ants)]).model_dump_json() + "\n")
            if env.steps % plan.checkpoint_every == 0 or env.done:
                save_actors(actors, directory, env.steps, include_memory=plan.respawn)
                snapshots.append(env.steps)
    for actor, before in zip(actors, frozen, strict=True):
        if isinstance(actor, OnlineForager):
            actor.check_frozen()
            if actor.agent.rewards or actor.agent.proposal is not None or actor.agent.awaiting_feedback:
                raise AssertionError("终止后仍有待处理反馈")
        elif isinstance(actor, NeuralForager) and any(not a.equal(b) for a, b in zip(
                before, [*actor.model.parameters(), *actor.motor.parameters()], strict=True)):
            raise AssertionError("冻结MLP改变")
    result = WorldResult(seed=seed, condition=condition.name, arm=arm, steps=env.steps,
        active_individual_steps=active_steps, deliveries=sum(a.deliveries for a in env.ants), pickups=sum(a.pickups for a in env.ants),
        budget_returns=sum(a.budget_returns for a in env.ants),
        deaths=int(env.deaths.sum()) if isinstance(env, RevivingColony) else int(env.killed.sum()),
        exhausted=int(env.terminations.sum()) if isinstance(env, RevivingColony) else sum(a.exhausted for a in env.ants),
        injury=float(env.total_injury.sum()) if isinstance(env, RevivingColony) else float(env.injuries.sum()),
        reward=reward, decisions=decisions, eligible=eligible, accepted=accepted,
        writes=[a.agent.writes if isinstance(a, OnlineForager) else 0 for a in actors], snapshots=snapshots,
        elapsed_seconds=time.perf_counter() - started,
        revivals=int(env.revivals.sum()) if isinstance(env, RevivingColony) else 0)
    (directory / "result.json").write_text(result.model_dump_json(indent=2))
    return result


def run(plan: ContinuousPlan, directory: Path, *, protocol_sha256: str, smoke: bool = False) -> None:
    if smoke:
        plan = ContinuousPlan.model_validate({**plan.model_dump(), "seeds": (18999,), "checkpoint_every": 4,
            "environment": plan.environment.model_copy(update={"ants": 2, "horizon": 8}),
            "adaptation": plan.adaptation.model_copy(update={"window": 4})})
    paths = [Path(plan.motor), *[plan.policy_path(i) for i in range(plan.environment.ants)],
             *[plan.gate_path(i) for i in range(plan.environment.ants)], *[plan.mlp_path(i) for i in range(plan.environment.ants)]]
    sources = [SourceRecord(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    execution = Execution(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started_at=datetime.now(timezone.utc).isoformat(), smoke=smoke, plan=plan, protocol_sha256=protocol_sha256, sources=sources)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (directory / "worlds.jsonl").open("x") as output:
        for seed in plan.seeds:
            for condition in plan.conditions:
                for arm in ARMS:
                    result = run_world(plan, seed, condition, arm, directory / f"{condition.name}-{seed}-{arm}")
                    output.write(result.model_dump_json() + "\n")
                    output.flush()
                    print(result.model_dump_json(), flush=True)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in sources):
        raise AssertionError("源参数文件改变")
    execution.completed_at = datetime.now(timezone.utc).isoformat()
    (directory / "execution.json").write_text(execution.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    content = args.config.read_bytes()
    run(ContinuousPlan.model_validate_json(content), args.output, protocol_sha256=hashlib.sha256(content).hexdigest(), smoke=args.smoke)
