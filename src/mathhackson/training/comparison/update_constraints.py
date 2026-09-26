"""既有四步轨迹的更新约束诊断；不生成新候选或推进世界。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import torch

from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution
from mathhackson.training.foraging.trust_candidate import TrustConfig, rotation_probabilities
from .acceptance import TapeStore
from .auditing import verify_manifest
from .continuous import ContinuousPlan, make_actors
from .online_actor import UpdateRecord
from .return_reward_audit import audit_world, frames
from .steering_audit import Change, MotorTable, Sample, local_away, measure
from .trigger_audit import audit_schedule


class ConstraintChange(BaseModel):
    effect: Change
    mean_kl: float
    maximum_kl: float
    mean_kl_fraction: float
    maximum_kl_fraction: float
    backtracks: int
    step_norm: float
    step_fraction: float
    residual_norm: float
    residual_fraction: float
    angle_change_degrees: float
    minimum_tanh_slope: float
    saturated_states: int
    feature_rank: int
    probability_jacobian_rank: int
    motor_total_variation: float


class Spread(BaseModel):
    minimum: float
    median: float
    maximum: float
    mean: float

    @classmethod
    def of(cls, values: list[float]) -> Spread:
        return cls(minimum=min(values), median=float(np.median(values)),
                   maximum=max(values), mean=float(np.mean(values)))


class WorldAudit(BaseModel):
    seed: int
    arm: str
    actions_verified: int
    memory_snapshots_verified: int
    decisions: int
    nonzero_proposals: int
    zero_proposals: int
    zero_backtrack_exhausted: int
    writes: int


class DistributionSummary(BaseModel):
    seed: int
    writes: int
    injured_writes: int
    direction_tv: Spread
    motor_tv: Spread
    absolute_move_change: Spread
    absolute_turn_change_degrees: Spread
    mean_kl_fraction: Spread
    maximum_kl_fraction: Spread
    step_fraction: Spread
    residual_fraction: Spread
    angle_change_degrees: Spread
    minimum_tanh_slope: Spread
    feature_rank: Spread
    probability_jacobian_rank: Spread
    backtracked_writes: int
    distribution_near_bound: int
    step_near_bound: int
    residual_near_bound: int
    saturated_writes: int
    motor_tv_below_one_percent: int


class Summary(BaseModel):
    files_verified: int
    worlds: list[WorldAudit]
    distributions: list[DistributionSummary]
    changes: list[ConstraintChange]


def probability_jacobian(weights: np.ndarray, bases: np.ndarray, features: np.ndarray) -> np.ndarray:
    """对旋转角和softmax求导，保留固定均匀混合项的零导数。"""
    z = features @ weights
    angle = np.pi * np.tanh(z)
    c, s = np.cos(angle), np.sin(angle)
    adjusted = np.stack((c * bases[:, 0] - s * bases[:, 1],
                         s * bases[:, 0] + c * bases[:, 1]), axis=-1)
    directions = DIRECTIONS.numpy().astype(np.float64)
    logits = 8. * adjusted @ directions.T
    softmax = np.exp(logits - logits.max(axis=-1, keepdims=True))
    softmax /= softmax.sum(axis=-1, keepdims=True)
    perpendicular = np.stack((-adjusted[:, 1], adjusted[:, 0]), axis=-1)
    score_slope = 8. * perpendicular @ directions.T
    probability_slope = .8 * softmax * (score_slope - (softmax * score_slope).sum(axis=-1, keepdims=True))
    angle_slope = np.pi * (1. - np.tanh(z) ** 2)[:, None] * features
    return probability_slope[:, :, None] * angle_slope[:, None, :]


def motor_variation(before: np.ndarray, after: np.ndarray, table: MotorTable) -> float:
    # 相同前进布尔值和实际转向量合并为同一个动作，不能用平均方向代替。
    _, groups = np.unique(np.stack((table.move, table.turn), axis=-1), axis=0, return_inverse=True)
    differences = np.stack([(after - before)[:, groups == group].sum(-1)
                            for group in range(int(groups.max()) + 1)], axis=-1)
    return float(np.abs(differences).sum(-1).mean() / 2.)


def measure_constraints(seed: int, record: UpdateRecord, samples: list[Sample], table: MotorTable,
                        config: TrustConfig, stable: np.ndarray) -> ConstraintChange:
    bases = torch.stack([sample.base for sample in samples])
    features = torch.stack([sample.features for sample in samples])
    before = rotation_probabilities(torch.tensor(record.before, dtype=torch.float32), bases, features)
    after = rotation_probabilities(torch.tensor(record.after, dtype=torch.float32), bases, features)
    # 批量矩阵乘法与逐状态点积的浮点32位求和顺序不同。
    probability_tolerance = 16. * np.finfo(np.float32).eps
    np.testing.assert_allclose(before.numpy(), np.stack([s.probabilities for s in samples]),
                               atol=probability_tolerance, rtol=probability_tolerance)
    divergence = (before * (before.log() - after.log())).sum(-1).clamp_min(0.)
    mean_kl, maximum_kl = float(divergence.mean()), float(divergence.max())
    if abs(mean_kl - record.diagnostics.mean_kl) > 2e-6 or abs(maximum_kl - record.diagnostics.maximum_kl) > 2e-6:
        raise ValueError('保存的分布约束与实际写入不符')
    old, new = np.asarray(record.before, np.float64), np.asarray(record.after, np.float64)
    feature_array, base_array = features.numpy().astype(np.float64), bases.numpy().astype(np.float64)
    z_before, z_after = feature_array @ old, feature_array @ new
    slopes = 1. - np.tanh(z_before) ** 2
    jacobian = probability_jacobian(old, base_array, feature_array).reshape(-1, old.size)
    step_norm, residual_norm = float(np.linalg.norm(new - old)), float(np.linalg.norm(new - stable))
    if step_norm > config.maximum_step_norm + 1e-5 or residual_norm > config.maximum_residual_norm + 1e-5:
        raise ValueError('实际参数越过已冻结约束')
    return ConstraintChange(effect=measure(seed, record, samples, table), mean_kl=mean_kl, maximum_kl=maximum_kl,
        mean_kl_fraction=mean_kl / config.maximum_mean_kl, maximum_kl_fraction=maximum_kl / config.maximum_state_kl,
        backtracks=record.diagnostics.backtracks, step_norm=step_norm, step_fraction=step_norm / config.maximum_step_norm,
        residual_norm=residual_norm, residual_fraction=residual_norm / config.maximum_residual_norm,
        angle_change_degrees=float(np.mean(np.abs(np.tanh(z_after) - np.tanh(z_before))) * 180.),
        minimum_tanh_slope=float(slopes.min()), saturated_states=int(np.sum(slopes <= .01)),
        feature_rank=int(np.linalg.matrix_rank(feature_array)),
        probability_jacobian_rank=int(np.linalg.matrix_rank(jacobian)),
        motor_total_variation=motor_variation(before.numpy(), after.numpy(), table))


def replay_world(store: TapeStore, seed: int, arm: str) -> tuple[WorldAudit, list[ConstraintChange]]:
    plan = store.execution.plan
    path, world = store.world('moving-danger', seed, arm)
    audit_world(path, world, plan)
    audit_schedule(path, world, plan)
    actors = make_actors(plan, seed, arm)
    records = [UpdateRecord.model_validate_json(line) for line in (path / 'updates.jsonl').read_text().splitlines()]
    indexed = {(record.tick, record.individual): record for record in records}
    stable = [actor.agent.weights().astype(np.float64) for actor in actors]
    tables = [MotorTable.from_motor(actor.agent.motor) for actor in actors]
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
                        raise ValueError('实际动作未重建一致')
                    pending[i].append(Sample(learner.policy(tensor, history)[0],
                        tensor[:72].reshape(9, 8)[:, 3:].flatten(), probabilities.numpy(), choice,
                        local_away(tensor.numpy()), ant.injury_delta > 0.))
                    learner.history.append(hidden)
                    actions += 1
                record = indexed.get((frame.tick, i))
                if record:
                    np.testing.assert_array_equal(learner.weights(), record.before)
                    if len(pending[i]) != record.proposal.steps:
                        raise ValueError('窗口动作数量不符')
                    if record.changed:
                        changes.append(measure_constraints(seed, record, pending[i], tables[i], plan.adaptation, stable[i]))
                    learner.assign_weights(np.asarray(record.after, dtype=np.float32))
                    pending[i].clear()
                if frame.tick in world.snapshots:
                    with np.load(path / f'tick-{frame.tick:04d}-ant-{i:02d}.memory.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(torch.stack(tuple(learner.history)).numpy(), saved['history'])
                        np.testing.assert_array_equal(learner.random.get_state().numpy(), saved['random_state'])
                    snapshots += 1
    if any(pending) or actions != world.active_individual_steps or len(changes) != sum(world.writes):
        raise ValueError('动作、窗口或写入数量不完整')
    nonzero = sum(any(record.proposal.delta) for record in records)
    result = WorldAudit(seed=seed, arm=arm, actions_verified=actions, memory_snapshots_verified=snapshots,
        decisions=len(records), nonzero_proposals=nonzero, zero_proposals=len(records) - nonzero,
        zero_backtrack_exhausted=sum(not any(r.proposal.delta) and r.diagnostics.backtracks == plan.adaptation.backtracks
                                    for r in records), writes=len(changes))
    return result, changes


def summarize(seed: int, rows: list[ConstraintChange]) -> DistributionSummary:
    def spread(name: str) -> Spread:
        return Spread.of([getattr(row, name) for row in rows])
    return DistributionSummary(seed=seed, writes=len(rows), injured_writes=sum(row.effect.injury_steps > 0 for row in rows),
        direction_tv=Spread.of([r.effect.probability_total_variation for r in rows]), motor_tv=spread('motor_total_variation'),
        absolute_move_change=Spread.of([abs(r.effect.move_probability_change) for r in rows]),
        absolute_turn_change_degrees=Spread.of([abs(r.effect.turn_degrees_change) for r in rows]),
        mean_kl_fraction=spread('mean_kl_fraction'), maximum_kl_fraction=spread('maximum_kl_fraction'),
        step_fraction=spread('step_fraction'), residual_fraction=spread('residual_fraction'),
        angle_change_degrees=spread('angle_change_degrees'), minimum_tanh_slope=spread('minimum_tanh_slope'),
        feature_rank=spread('feature_rank'), probability_jacobian_rank=spread('probability_jacobian_rank'),
        backtracked_writes=sum(r.backtracks > 0 for r in rows),
        distribution_near_bound=sum(max(r.mean_kl_fraction, r.maximum_kl_fraction) >= .95 for r in rows),
        step_near_bound=sum(r.step_fraction >= .95 for r in rows), residual_near_bound=sum(r.residual_fraction >= .95 for r in rows),
        saturated_writes=sum(r.saturated_states > 0 for r in rows),
        motor_tv_below_one_percent=sum(r.motor_total_variation < .01 for r in rows))


def audit(directory: Path, manifest: Path) -> Summary:
    files = verify_manifest(directory, manifest)
    store = TapeStore(directory / 'four')
    protocol = Path('.research/protocols/four-frame-development.json').read_bytes()
    expected = ContinuousPlan.model_validate_json(protocol)
    if store.execution.plan != expected or store.execution.protocol_sha256 != hashlib.sha256(protocol).hexdigest():
        raise ValueError('原始协议与冻结四步配置不符')
    for source in store.execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError('原始模型身份改变')
    worlds, changes = [], []
    for seed in expected.seeds:
        for arm in ('skip', 'always'):
            world, rows = replay_world(store, seed, arm)
            worlds.append(world)
            changes.extend(rows)
    return Summary(files_verified=files, worlds=worlds, changes=changes,
                   distributions=[summarize(seed, [row for row in changes if row.effect.seed == seed]) for seed in expected.seeds])
