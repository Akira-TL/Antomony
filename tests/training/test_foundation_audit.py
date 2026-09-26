import gzip

import pytest

from mathhackson.training.comparison.qualification import QualificationPlan
from mathhackson.training.comparison.run import Frame, WorldRecord
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.comparison.auditing import audit_world


def sample(tmp_path):
    plan = QualificationPlan(environment=ColonyConfig(ants=1))
    row = WorldRecord(model="rules", seed=1, sampled=None, task="food", steps=2, deliveries=1, pickups=1,
                      budget_returns=0, returning_individuals=0, exhausted=0, mean_max_radius=1., updates=[0])
    with gzip.open(tmp_path / "rules-food-1-None.jsonl.gz", "wt") as stream:
        for tick, carrying in ((1, True), (2, False)):
            stream.write(Frame(tick=tick, moves=[True], turns=[0.], positions=[[1., 0.]],
                               carrying=[carrying], rewards=[1. if carrying else 4.]).model_dump_json() + "\n")
    return plan, row


def test_foundation_audit_reconstructs_delivery_and_radius(tmp_path):
    plan, row = sample(tmp_path)
    assert audit_world(tmp_path, row, plan.environment) == 2


@pytest.mark.parametrize("field,value", [("deliveries", 0), ("mean_max_radius", 2.), ("steps", 3), ("updates", [1])])
def test_foundation_audit_rejects_inconsistent_world(tmp_path, field, value):
    plan, row = sample(tmp_path)
    with pytest.raises(ValueError):
        audit_world(tmp_path, row.model_copy(update={field: value}), plan.environment)


def test_training_audit_requires_explicit_opt_in(tmp_path):
    plan, row = sample(tmp_path)
    row.updates = [3]
    assert audit_world(tmp_path, row, plan.environment, frozen=False) == 2
    with pytest.raises(ValueError):
        audit_world(tmp_path, row, plan.environment)
