"""四步判断下的动作保留配对审计；不训练或筛选接受器。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.foraging.candidate_probe import CandidateRecord, Execution, ParentFrame, ProbePlan, WorldRecord
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig
from .auditing import verify_manifest


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: Path
    manifest: Path
    output: Path


class Difference(BaseModel):
    positive: int
    negative: int
    tied: int
    mean: float | None


def describe(values: list[float]) -> Difference:
    a = np.asarray(values)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("差值必须是一维有限值")
    return Difference(positive=int((a > 1e-6).sum()), negative=int((a < -1e-6).sum()),
                      tied=int((np.abs(a) <= 1e-6).sum()), mean=float(a.mean()) if len(a) else None)


class CandidateScore(BaseModel):
    reward: Difference
    focal_deliveries: Difference
    colony_deliveries: Difference
    nonzero_candidates: int


def score(rows: list[CandidateRecord]) -> CandidateScore:
    return CandidateScore(
        reward=describe([r.result.accept.focal_reward - r.result.skip.focal_reward for r in rows]),
        focal_deliveries=describe([float(r.result.accept.focal_deliveries - r.result.skip.focal_deliveries) for r in rows]),
        colony_deliveries=describe([float(r.result.accept.colony_deliveries - r.result.skip.colony_deliveries) for r in rows]),
        nonzero_candidates=sum(r.result.changed for r in rows))


class Pair(BaseModel):
    seed: int
    count: int
    four: CandidateScore
    sixteen: CandidateScore
    reward_difference: Difference
    delivery_difference: Difference


class Result(BaseModel):
    files_verified: int
    snapshots_verified: int
    frames_verified: int
    pairs: list[Pair]
    parents_identical: bool
    development_continue: bool


def decide(pairs: list[Pair]) -> bool:
    if len(pairs) != 2 or len({p.seed for p in pairs}) != 2:
        raise ValueError("判定需要两个不同世界")
    if any(not p.count or p.reward_difference.mean is None or p.delivery_difference.mean is None for p in pairs):
        return False
    return (all(p.sixteen.reward.positive > 0 and p.sixteen.reward.positive >= p.four.reward.positive
                and p.reward_difference.mean >= 0. and p.delivery_difference.mean >= 0. for p in pairs)
            and any(p.reward_difference.mean > 0. for p in pairs)
            and any(p.sixteen.focal_deliveries.positive > 0 for p in pairs))


def verify_pair(left: list[CandidateRecord], right: list[CandidateRecord], lp: str, rp: str) -> None:
    if lp != rp or len(left) != len(right):
        raise ValueError("两种配置的父轨迹或配对数量不同")
    for first, second in zip(left, right, strict=True):
        if ((first.tick, first.focal, first.observation, first.hidden, first.result.skip)
                != (second.tick, second.focal, second.observation, second.hidden, second.result.skip)):
            raise ValueError("配对不在相同状态或随机延续上")
        if first.action_effects is None or second.action_effects is None:
            raise ValueError("动作特征缺失")
        before = (*range(16), 32, 34)
        if any(first.action_effects.values[i] != second.action_effects.values[i] for i in before):
            raise ValueError("修改前动作分布不匹配")


def audit_world(folder: Path, plan: ProbePlan, world: WorldRecord) -> tuple[list[CandidateRecord], int, str]:
    if world.condition != "benign" or world.deaths or not 1 <= world.steps <= plan.environment.horizon:
        raise ValueError("世界条件或时限错误")
    with gzip.open(folder / "parent.jsonl.gz", "rt") as stream:
        parent = stream.read()
    frames = [ParentFrame.model_validate_json(line) for line in parent.splitlines()]
    if len(frames) != world.steps:
        raise ValueError("父轨迹不完整")
    for tick, frame in enumerate(frames, 1):
        if (frame.tick != tick or any(len(v) != plan.environment.ants for v in (frame.moves, frame.turns, frame.rewards))
                or not np.isfinite(frame.turns + frame.rewards).all()):
            raise ValueError("父轨迹帧无效")
    snapshots = 0
    initial_weights = []
    for focal in range(plan.environment.ants):
        initial = np.zeros((9, 5), dtype=np.float32)
        initial[:, 0] = np.random.default_rng(world.seed * 8 + focal).choice((-1., 1.)) * .25
        initial_weights.append(initial.flatten())
        base = MemoryPolicy.load(plan.policy_path(focal)).state_dict()
        ticks = set(range(64, world.steps + 1, 64)) | {0, world.steps}
        paths = {folder / f"tick-{t:04d}-ant-{focal:02d}.residual.npz" for t in ticks}
        if set(folder.glob(f"*-ant-{focal:02d}.residual.npz")) != paths:
            raise ValueError("快照时点缺失或额外")
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                if (str(saved["adapter_kind"]) != "direction-trust" or int(saved["writes"]) != 0
                        or np.any(saved["stable"]) or not np.array_equal(saved["fast"], initial.flatten())
                        or TrustConfig.model_validate_json(str(saved["config_json"])) != plan.adaptation):
                    raise ValueError("父轨迹残差或配置改变")
            state = MemoryPolicy.load(path.with_name(path.name.replace(".residual.npz", ".npz"))).state_dict()
            if any(not torch.equal(state[key], value) for key, value in base.items()):
                raise ValueError("基础或记忆参数改变")
            snapshots += 1
    rows = [CandidateRecord.model_validate_json(line) for line in (folder / "pairs.jsonl").read_text().splitlines()]
    if len(rows) != world.pairs or len(rows) > plan.pairs_per_world or len({(r.tick, r.focal) for r in rows}) != len(rows):
        raise ValueError("候选数量或身份错误")
    for row in rows:
        if (row.seed != world.seed or row.condition != "benign" or row.tick % 16 or not 0 < row.tick < world.steps
                or not 0 <= row.focal < plan.environment.ants or row.proposal.steps != 4
                or row.diagnostics is None or row.action_effects is None
                or row.diagnostics.credit_steps != plan.adaptation.credit_horizon
                or row.diagnostics.mean_kl > plan.adaptation.maximum_mean_kl + 1e-6
                or row.diagnostics.maximum_kl > plan.adaptation.maximum_state_kl + 1e-6):
            raise ValueError("候选时序、归因范围或分布约束错误")
        if (len(row.observation) != 78 or len(row.hidden) != 8 or len(row.proposal.delta) != 45
                or not np.isfinite([*row.observation, *row.hidden, *row.proposal.delta]).all()):
            raise ValueError("局部记录维度或有限性错误")
        initial = initial_weights[row.focal]
        expected = initial + np.asarray(row.proposal.delta, dtype=np.float32)
        if (not np.array_equal(expected, row.result.accepted_weights)
                or row.result.changed != bool(np.any(expected != initial))
                or not row.result.changed and row.result.accept != row.result.skip):
            raise ValueError("真实分支写入与提案不一致")
        for outcome in (row.result.accept, row.result.skip):
            if outcome.colony_deaths or outcome.focal_injury or not 0 < outcome.steps <= plan.branch_horizon:
                raise ValueError("分支违反无危险范围或时限")
    return rows, snapshots, parent


def run(config: Config) -> Result:
    files = verify_manifest(config.input_directory, config.manifest)
    variants = ("four", "sixteen")
    executions = [Execution.model_validate_json((config.input_directory / v / "execution.json").read_text()) for v in variants]
    a, b = executions
    if (a.source_commit != b.source_commit or a.sources != b.sources
            or a.plan != b.plan.model_copy(update={"adaptation": b.plan.adaptation.model_copy(update={"credit_horizon": None})})):
        raise ValueError("配置不是仅改变动作保留范围")
    for variant, execution, horizon in zip(variants, executions, (None, 16), strict=True):
        protocol = Path(f".research/protocols/credit-history-{variant}.json")
        plan = execution.plan
        if (execution.smoke or plan != ProbePlan.model_validate_json(protocol.read_text())
                or execution.plan_sha256 != hashlib.sha256(protocol.read_bytes()).hexdigest()
                or plan.adaptation.window != 4 or plan.adaptation.credit_horizon != horizon
                or plan.sample_interval != 16 or not plan.include_zero_candidates
                or plan.initial_residual_norm != .75 or plan.initial_residual_pattern != "coherent-fourth"
                or plan.policy_kind != "mlp-memory"):
            raise ValueError("执行身份不符合冻结协议")
        paths = {plan.policy_path(i) for i in range(plan.environment.ants)} | {Path(plan.motor_path)}
        if len(paths) != len(execution.sources) or {Path(s.path) for s in execution.sources} != paths:
            raise ValueError("来源模型缺失或重复")
        if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in execution.sources):
            raise ValueError("来源参数散列改变")
    worlds = [[WorldRecord.model_validate_json(line) for line in (config.input_directory / v / "worlds.jsonl").read_text().splitlines()]
              for v in variants]
    if any([w.seed for w in group] != list(a.plan.seeds) for group in worlds):
        raise ValueError("世界不完整")
    snapshots = frames = 0
    pairs = []
    for old, new in zip(*worlds, strict=True):
        if (old.seed, old.steps, old.pairs, old.deliveries, old.deaths) != (new.seed, new.steps, new.pairs, new.deliveries, new.deaths):
            raise ValueError("父轨迹汇总不匹配")
        checked = [audit_world(config.input_directory / v / f"benign-{world.seed}", execution.plan, world)
                   for v, execution, world in zip(variants, executions, (old, new), strict=True)]
        (left, x, lp), (right, y, rp) = checked
        verify_pair(left, right, lp, rp)
        snapshots += x + y
        frames += old.steps + new.steps
        pairs.append(Pair(seed=old.seed, count=len(left), four=score(left), sixteen=score(right),
            reward_difference=describe([v.result.accept.focal_reward - u.result.accept.focal_reward for u, v in zip(left, right, strict=True)]),
            delivery_difference=describe([float(v.result.accept.focal_deliveries - u.result.accept.focal_deliveries) for u, v in zip(left, right, strict=True)])))
    result = Result(files_verified=files, snapshots_verified=snapshots, frames_verified=frames,
                    pairs=pairs, parents_identical=True, development_continue=decide(pairs))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result
