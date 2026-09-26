import pytest

from mathhackson.training.comparison.survival_audit import Contrast, decide


def rows():
    return [Contrast(seed=seed, comparator=comparator, failures=-1, deaths=-2,
                     exhaustion=1, deliveries=-20, injury=-2.)
            for seed in (19501, 19502) for comparator in ('legacy-always', 'skip')]


def test_survival_priority_allows_delivery_loss_but_not_more_failures():
    values = rows()
    assert decide(values)
    values[0].failures = 1
    assert not decide(values)
    values[0].deliveries = 10000
    assert not decide(values)


def test_both_comparators_need_some_strict_survival_improvement():
    values = rows()
    for row in values:
        if row.comparator == 'skip':
            row.failures = 0
    assert not decide(values)
    with pytest.raises(ValueError):
        decide(values[:-1])
    values[-1] = values[0]
    with pytest.raises(ValueError):
        decide(values)
