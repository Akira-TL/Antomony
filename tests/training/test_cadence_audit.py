from pathlib import Path

import pytest

from mathhackson.training.comparison.cadence_audit import Contrast, decide, validate_pair
from mathhackson.training.comparison.continuous import ContinuousPlan


def test_only_window_changes_in_registered_pair():
    root = Path(__file__).resolve().parents[2] / '.research/protocols'
    old = ContinuousPlan.model_validate_json((root / 'window-cadence-16.json').read_bytes())
    new = ContinuousPlan.model_validate_json((root / 'four-frame-development.json').read_bytes())
    validate_pair(old, new)
    with pytest.raises(ValueError, match='仅改变'):
        validate_pair(old, new.model_copy(update={'adaptation': new.adaptation.model_copy(update={'learning_rate': .01})}))
    with pytest.raises(ValueError, match='仅改变'):
        validate_pair(old, new.model_copy(update={'adaptation': new.adaptation.model_copy(update={'feedback_trigger': 'negative-feedback'})}))


def contrasts():
    return [Contrast(seed=s, comparator=c, failures=-1, deaths=-2, exhaustion=1, deliveries=-20)
            for s in (19701, 19702) for c in ('sixteen-always', 'skip')]


def test_success_needs_both_comparators_and_does_not_trade_food_for_survival():
    rows = contrasts()
    assert decide(rows)
    rows[0].failures = 1
    rows[0].exhaustion = 3
    rows[0].deliveries = 100
    assert not decide(rows)
    rows = contrasts()
    for row in rows:
        if row.comparator == 'skip':
            row.failures = row.deaths = row.exhaustion = 0
    assert not decide(rows)


def test_reject_incomplete_duplicated_or_inconsistent_results():
    rows = contrasts()
    with pytest.raises(ValueError, match='完整'):
        decide(rows[:-1])
    with pytest.raises(ValueError, match='重复'):
        decide([*rows[:-1], rows[0]])
    rows[0].failures = -100
    with pytest.raises(ValueError, match='分量'):
        decide(rows)
