import pytest

from mathhackson.training.comparison.continuous import ARMS, Condition, ContinuousPlan, WorldResult
from mathhackson.training.comparison.continuous_scores import summarize
from mathhackson.training.foraging.disturbance import DisturbanceConfig


def inputs():
    plan = ContinuousPlan(conditions=tuple(Condition(name=name, disturbance=DisturbanceConfig())
        for name in ("reference", "slow", "periodic", "moving-danger")))
    rows = [WorldResult(seed=seed, condition=c.name, arm=arm, steps=768, active_individual_steps=6144,
        deliveries=12 if arm == "learned" and c.name != "reference" else 10, pickups=16, budget_returns=0, deaths=0, exhausted=0,
        injury=0., reward=0., decisions=2, eligible=int(c.name != "reference"), accepted=int(arm == "learned" and c.name != "reference"),
        writes=[int(arm == "learned" and c.name != "reference"), *[0] * 7], snapshots=[0,768], elapsed_seconds=1.)
        for seed in plan.seeds for c in plan.conditions for arm in ARMS]
    return plan, rows


def test_paired_scores_are_equal_world_weight_not_survivor_weight():
    plan, rows = inputs()
    rows[5].active_individual_steps = 1
    report = summarize(list(reversed(rows)), plan)
    assert report.passing_seeds == 4 and report.passing_conditions == 3 and report.development_continue
    assert all(r.deliveries == 2. for r in report.complex_seed_contrasts)
    assert not report.general_advantage_established


@pytest.mark.parametrize("failure", ["mortality", "exhaustion", "no-write", "two-seeds", "one-condition", "negative-mean"])
def test_every_fixed_development_guard_can_reject(failure):
    plan, rows = inputs()
    for row in rows:
        if row.arm != "learned" or row.condition == "reference":
            continue
        if failure == "mortality":
            row.deaths = 1
        elif failure == "exhaustion":
            row.exhausted = 1
        elif failure == "no-write":
            row.writes = [0] * 8
        elif failure == "two-seeds" and row.seed in plan.seeds[:2]:
            row.deliveries = 9
        elif failure == "one-condition" and row.condition != "slow":
            row.deliveries = 10
        elif failure == "negative-mean" and row.seed == plan.seeds[0]:
            row.deliveries = 0
    report = summarize(rows, plan)
    assert not report.development_continue
    if failure == "negative-mean":
        assert report.passing_seeds == 3
        assert any(r.deliveries < 0 for r in report.complex_seed_contrasts)


def test_missing_or_duplicate_world_cannot_be_treated_as_zero():
    plan, rows = inputs()
    with pytest.raises(ValueError):
        summarize(rows[:-1], plan)
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        summarize(rows, plan)


def test_invalid_reference_cannot_support_development_success():
    plan, rows = inputs()
    rows[0].deliveries += 1
    with pytest.raises(ValueError, match="正常参照"):
        summarize(rows, plan)
