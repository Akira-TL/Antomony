"""只读重建方向采样，区分期望方向与冻结动作的实际概率变化。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.direction.environment import MAX_TURN, STEP_DISTANCE
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution
from mathhackson.training.foraging.trust_candidate import rotation_probabilities
from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ContinuousPlan, make_actors
from .online_actor import OnlineForager, UpdateRecord
from .return_reward_audit import audit_world, frames
from .trigger_audit import audit_schedule


class Effect(BaseModel):
    move_probability: float
    mean_turn_degrees: float
    desired_away: float | None
    heading_away: float | None
    motion_away: float | None


class Change(BaseModel):
    seed: int
    individual: int
    tick: int
    steps: int
    injury_steps: int
    terminal: bool
    gradient_norm: float
    probability_total_variation: float
    chosen_probability_change: float
    move_probability_change: float
    turn_degrees_change: float
    gradient_states: int
    desired_away_change: float | None
    heading_away_change: float | None
    motion_away_change: float | None


class WorldSummary(BaseModel):
    seed: int
    arm: str
    active_actions_verified: int
    memory_snapshots_verified: int
    writes: int
    injured_writes: int
    injured_gradient_writes: int
    heading_more_away: int
    heading_less_away: int
    desired_better_heading_worse: int
    reduced_move_probability: int
    mean_move_probability_change: float | None
    mean_heading_away_change: float | None


class Summary(BaseModel):
    files_verified: int
    worlds: list[WorldSummary]
    changes: list[Change]


@dataclass(frozen=True)
class MotorTable:
    move: np.ndarray
    turn: np.ndarray
    headings: np.ndarray

    @classmethod
    def from_motor(cls, motor: DirectionMotor) -> MotorTable:
        actions = [motor.decide(direction.numpy()) for direction in DIRECTIONS]
        turn = np.asarray([action.turn for action in actions]) * MAX_TURN
        return cls(np.asarray([action.move for action in actions], dtype=float), turn,
                   np.stack((np.cos(turn), np.sin(turn)), axis=-1))


def local_away(observation: np.ndarray) -> np.ndarray | None:
    # 只作事后局部方向参照，不作为控制器输入或无碰撞路径答案。
    responses = np.expm1(observation[:72].reshape(9, 8).astype(np.float64) * math.log(9.))[:, 3:].sum(-1)
    gradient = np.asarray([responses[1] - responses[3], responses[2] - responses[4]])
    length = float(np.linalg.norm(gradient))
    return -gradient / length if length > 1e-6 else None


def expected_effect(probabilities: np.ndarray, table: MotorTable, away: np.ndarray | None) -> Effect:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (16,) or not np.isfinite(p).all() or np.any(p < 0.) or not np.isclose(p.sum(), 1., atol=1e-6):
        raise ValueError('方向概率分布无效')
    if away is not None and (away.shape != (2,) or not np.isfinite(away).all() or not np.isclose(np.linalg.norm(away), 1.)):
        raise ValueError('局部方向参照无效')
    return Effect(move_probability=float(p @ table.move), mean_turn_degrees=math.degrees(float(p @ table.turn)),
        desired_away=float(p @ DIRECTIONS.numpy() @ away) if away is not None else None,
        heading_away=float(p @ table.headings @ away) if away is not None else None,
        motion_away=float((p * table.move) @ table.headings @ away) * STEP_DISTANCE if away is not None else None)


@dataclass(frozen=True)
class Sample:
    base: torch.Tensor
    features: torch.Tensor
    probabilities: np.ndarray
    choice: int
    away: np.ndarray | None
    injury: bool


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def measure(seed: int, record: UpdateRecord, samples: list[Sample], table: MotorTable) -> Change:
    after = rotation_probabilities(torch.tensor(record.after, dtype=torch.float32),
        torch.stack([s.base for s in samples]), torch.stack([s.features for s in samples])).numpy()
    before = np.stack([s.probabilities for s in samples])
    pairs = [(expected_effect(a, table, s.away), expected_effect(b, table, s.away))
             for s, a, b in zip(samples, before, after, strict=True)]
    def difference(name: str) -> float | None:
        return mean_or_none([getattr(b, name) - getattr(a, name) for a, b in pairs if getattr(a, name) is not None])
    choices = np.asarray([s.choice for s in samples])
    return Change(seed=seed, individual=record.individual, tick=record.tick, steps=len(samples),
        injury_steps=sum(s.injury for s in samples), terminal=record.terminal, gradient_norm=record.proposal.gradient_norm,
        probability_total_variation=float(np.abs(after - before).sum(-1).mean() / 2.),
        chosen_probability_change=float((after - before)[np.arange(len(samples)), choices].mean()),
        move_probability_change=difference('move_probability'), turn_degrees_change=difference('mean_turn_degrees'),
        gradient_states=sum(s.away is not None for s in samples), desired_away_change=difference('desired_away'),
        heading_away_change=difference('heading_away'), motion_away_change=difference('motion_away'))


def summarize(seed: int, arm: str, actions: int, snapshots: int, changes: list[Change]) -> WorldSummary:
    injured = [row for row in changes if row.injury_steps]
    directed = [row for row in injured if row.heading_away_change is not None]
    tolerance = 1e-7
    return WorldSummary(seed=seed, arm=arm, active_actions_verified=actions, memory_snapshots_verified=snapshots,
        writes=len(changes), injured_writes=len(injured), injured_gradient_writes=len(directed),
        heading_more_away=sum(row.heading_away_change > tolerance for row in directed),
        heading_less_away=sum(row.heading_away_change < -tolerance for row in directed),
        desired_better_heading_worse=sum(row.desired_away_change > tolerance and row.heading_away_change < -tolerance
                                       for row in directed),
        reduced_move_probability=sum(row.move_probability_change < -tolerance for row in injured),
        mean_move_probability_change=mean_or_none([row.move_probability_change for row in injured]),
        mean_heading_away_change=mean_or_none([row.heading_away_change for row in directed]))


def audit_world_steering(store: TapeStore, seed: int, arm: str) -> tuple[WorldSummary, list[Change]]:
    plan = store.execution.plan
    path, world = store.world('moving-danger', seed, arm)
    audit_world(path, world, plan)
    audit_schedule(path, world, plan)
    actors = make_actors(plan, seed, arm)
    if not all(isinstance(actor, OnlineForager) for actor in actors):
        raise ValueError('动作重建只用于在线或全部跳过组')
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    indexed = {(row.tick, row.individual): row for row in records}
    table = MotorTable.from_motor(actors[0].agent.motor)
    pending: list[list[Sample]] = [[] for _ in actors]
    changes = []
    actions = snapshots = 0
    with torch.inference_mode():
        for frame in frames(path):
            for i, ant in enumerate(frame.ants):
                learner = actors[i].agent
                if ant.active:
                    tensor = torch.tensor(ant.observation, dtype=torch.float32)
                    history = tuple(learner.history)
                    direction, hidden, _ = learner.predict(tensor, history)
                    probabilities = direction_distribution(direction).probs
                    choice = int(torch.multinomial(probabilities, 1, generator=learner.random))
                    action = learner.motor.decide(DIRECTIONS[choice].numpy())
                    if action.move != ant.move or abs(action.turn - ant.turn) > 1e-7:
                        raise ValueError('随机动作无法从原始权重与历史重建')
                    base = learner.policy(tensor, history)[0]
                    learner.history.append(hidden)
                    pending[i].append(Sample(base, tensor[:72].reshape(9, 8)[:, 3:].flatten(),
                        probabilities.numpy(), choice, local_away(tensor.numpy()), ant.injury_delta > 0.))
                    actions += 1
                record = indexed.get((frame.tick, i))
                if record:
                    np.testing.assert_array_equal(learner.weights(), record.before)
                    if len(pending[i]) != record.proposal.steps:
                        raise ValueError('重建窗口与原始提案不符')
                    if record.changed:
                        changes.append(measure(seed, record, pending[i], table))
                    learner.assign_weights(np.asarray(record.after, dtype=np.float32))
                    pending[i].clear()
                if frame.tick in world.snapshots:
                    with np.load(path / f'tick-{frame.tick:04d}-ant-{i:02d}.memory.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(torch.stack(tuple(learner.history)).numpy(), saved['history'])
                        np.testing.assert_array_equal(learner.random.get_state().numpy(), saved['random_state'])
                    snapshots += 1
    if any(pending) or actions != world.active_individual_steps or len(changes) != sum(world.writes):
        raise ValueError('动作、反馈或写入未完整重建')
    return summarize(seed, arm, actions, snapshots, changes), changes


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    store = TapeStore(directory / 'four')
    protocol = Path('.research/protocols/four-frame-development.json').read_bytes()
    expected = ContinuousPlan.model_validate_json(protocol)
    if store.execution.plan != expected or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest():
        raise ValueError('不匹配已冻结的四步配置')
    for source in store.execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError('重建所用源参数改变')
    worlds, changes = [], []
    for seed in expected.seeds:
        for arm in ('skip', 'always'):
            world, rows = audit_world_steering(store, seed, arm)
            worlds.append(world)
            changes.extend(rows)
    return Summary(files_verified=files, worlds=worlds, changes=changes)
