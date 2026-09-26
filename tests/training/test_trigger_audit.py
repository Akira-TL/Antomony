import pytest

from test_continuous import plan
from mathhackson.training.comparison.continuous import Condition, ContinuousPlan, run_world
from mathhackson.training.comparison.trigger_audit import Contrast, audit_schedule, decide
from mathhackson.training.foraging.disturbance import DisturbanceConfig


@pytest.mark.parametrize('trigger', ['window', 'negative-feedback'])
def test_actual_schedule_is_audited_and_missing_proposal_rejected(plan, tmp_path, trigger):
    condition = Condition(name='moving-danger', disturbance=DisturbanceConfig(contact_radius=20., injury_per_step=.08))
    plan = ContinuousPlan.model_validate({**plan.model_dump(), 'arms': ('skip', 'always'),
        'feedback_profile': 'survival-v1', 'respawn': True,
        'adaptation': {**plan.adaptation.model_dump(), 'feedback_trigger': trigger}})
    path = tmp_path / 'world'
    world = run_world(plan, 1, condition, 'always', path)
    assert audit_schedule(path, world, plan) == world.decisions
    records = (path / 'updates.jsonl').read_text().splitlines()
    (path / 'updates.jsonl').write_text('\n'.join(records[1:]) + '\n')
    with pytest.raises(ValueError, match='时序'):
        audit_schedule(path, world, plan)


def test_failure_priority_and_both_comparators_are_required():
    rows = [Contrast(seed=s, comparator=c, failures=-1, deaths=-2, exhaustion=1, deliveries=-20)
            for s in (19601, 19602) for c in ('window-always', 'skip')]
    assert decide(rows)
    rows[0].failures = 1
    assert not decide(rows)
    with pytest.raises(ValueError):
        decide(rows[:-1])
