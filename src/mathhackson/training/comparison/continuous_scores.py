"""按世界和条件等权描述完整部署后果，不用个体或步数扩大样本。"""
from __future__ import annotations

from pydantic import BaseModel

from .auditing import Estimate, describe
from .continuous import ARMS, ContinuousPlan, WorldResult


class Group(BaseModel):
    condition: str
    arm: str
    deliveries: Estimate
    deaths: Estimate
    exhausted: Estimate
    reward: Estimate
    active_individual_steps: Estimate
    decisions: int
    eligible: int
    accepted: int
    writes: int
    elapsed_seconds: Estimate


class Contrast(BaseModel):
    condition: str
    comparator: str
    deliveries: Estimate
    deaths: Estimate
    exhausted: Estimate


class SeedContrast(BaseModel):
    seed: int
    comparator: str
    deliveries: float
    deaths: float
    exhausted: float


class Summary(BaseModel):
    groups: list[Group]
    learned_minus_comparator: list[Contrast]
    complex_seed_contrasts: list[SeedContrast]
    passing_seeds: int
    passing_conditions: int
    mean_delivery_improvement: bool
    survival_not_worse: bool
    actual_learned_writes: int
    development_continue: bool
    general_advantage_established: bool = False


def summarize(worlds: list[WorldResult], plan: ContinuousPlan) -> Summary:
    conditions = tuple(condition.name for condition in plan.conditions)
    expected = {(seed, condition, arm) for seed in plan.seeds for condition in conditions for arm in ARMS}
    by_id = {(r.seed, r.condition, r.arm): r for r in worlds}
    if len(worlds) != len(expected) or set(by_id) != expected:
        raise ValueError("连续世界身份重复、缺失或多余")
    if set(conditions) != {"reference", "slow", "periodic", "moving-danger"}:
        raise ValueError("固定判据需要一个正常及三个复杂条件")
    for seed in plan.seeds:
        reference = [by_id[seed, "reference", arm] for arm in ("learned", "skip", "always")]
        first = reference[0].model_dump(exclude={"arm", "elapsed_seconds"})
        if any(r.model_dump(exclude={"arm", "elapsed_seconds"}) != first
               or any(r.writes) or r.accepted or r.eligible for r in reference):
            raise ValueError("正常参照三组后果不一致或发生更新")
    complex_conditions = tuple(c for c in conditions if c != "reference")
    groups, contrasts = [], []
    for condition in conditions:
        for arm in ARMS:
            rows = [by_id[seed, condition, arm] for seed in plan.seeds]
            groups.append(Group(condition=condition, arm=arm, deliveries=describe([float(r.deliveries) for r in rows]),
                deaths=describe([float(r.deaths) for r in rows]), exhausted=describe([float(r.exhausted) for r in rows]),
                reward=describe([r.reward for r in rows]), active_individual_steps=describe([float(r.active_individual_steps) for r in rows]),
                decisions=sum(r.decisions for r in rows), eligible=sum(r.eligible for r in rows),
                accepted=sum(r.accepted for r in rows), writes=sum(sum(r.writes) for r in rows),
                elapsed_seconds=describe([r.elapsed_seconds for r in rows])))
        for comparator in ARMS[1:]:
            pairs = [(by_id[seed, condition, "learned"], by_id[seed, condition, comparator]) for seed in plan.seeds]
            contrasts.append(Contrast(condition=condition, comparator=comparator,
                deliveries=describe([float(a.deliveries - b.deliveries) for a, b in pairs]),
                deaths=describe([float(a.deaths - b.deaths) for a, b in pairs]),
                exhausted=describe([float(a.exhausted - b.exhausted) for a, b in pairs])))
    seed_contrasts = []
    for seed in plan.seeds:
        for comparator in ("skip", "always"):
            pairs = [(by_id[seed, c, "learned"], by_id[seed, c, comparator]) for c in complex_conditions]
            seed_contrasts.append(SeedContrast(seed=seed, comparator=comparator,
                deliveries=describe([float(a.deliveries - b.deliveries) for a, b in pairs]).mean,
                deaths=describe([float(a.deaths - b.deaths) for a, b in pairs]).mean,
                exhausted=describe([float(a.exhausted - b.exhausted) for a, b in pairs]).mean))
    passing_seeds = sum(all(r.deliveries > 0. for r in seed_contrasts if r.seed == seed) for seed in plan.seeds)
    passing_conditions = sum(all(r.deliveries.mean > 0. for r in contrasts
                                if r.condition == c and r.comparator in ("skip", "always")) for c in complex_conditions)
    improves = all(describe([r.deliveries for r in seed_contrasts if r.comparator == c]).mean > 0. for c in ("skip", "always"))
    safe = all(describe([r.deaths for r in seed_contrasts if r.comparator == c]).mean <= 0.
               and describe([r.exhausted for r in seed_contrasts if r.comparator == c]).mean <= 0.
               for c in ("skip", "always"))
    writes = sum(sum(r.writes) for r in worlds if r.arm == "learned" and r.condition in complex_conditions)
    return Summary(groups=groups, learned_minus_comparator=contrasts, complex_seed_contrasts=seed_contrasts,
        passing_seeds=passing_seeds, passing_conditions=passing_conditions, mean_delivery_improvement=improves,
        survival_not_worse=safe, actual_learned_writes=writes,
        development_continue=(len(plan.seeds) == 4 and passing_seeds >= 3 and passing_conditions >= 2 and improves and safe and writes > 0))
