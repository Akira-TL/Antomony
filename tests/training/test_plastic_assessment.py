import pytest

from mathhackson.training.foraging.plastic_course.assessment import assess
from mathhackson.training.foraging.plastic_course.run import EvaluationRow


def rows():
    return [EvaluationRow(initialization=1701, condition=condition, batch_seed=62, mode=mode,
        mean_after_feedback=[.2 if mode == "learned" else .1] * 2,
        mean_last_quarter=[.25 if mode == "learned" else .1] * 2,
        acceptances=[12, 12], nonzero_writes=[12, 12], total_variation_from_initial=[.1, .1])
        for condition in ("reference", "association", "transient")
        for mode in ("learned", "off", "always", "matched")]


def test_success_requires_all_distinct_conditions_and_mixed_acceptance():
    result = assess(rows(), initialization=1701, steps=96)
    assert result.passed and result.accepted_fraction == .5
    assert result.contrasts[0].mean_difference == pytest.approx(.1)
    always = [row.model_copy(update={"acceptances": [24, 24]}) for row in rows()]
    result = assess(always, initialization=1701, steps=96)
    assert result.association_gain and not result.mixed_control and not result.passed


def test_negative_differences_survive_and_cannot_pass_on_other_conditions():
    items = [row.model_copy(update={"mean_after_feedback": [-.3, -.2]})
        if row.mode == "learned" and row.condition == "association" else row for row in rows()]
    result = assess(items, initialization=1701, steps=96)
    assert not result.passed and not result.association_gain
    assert next(c for c in result.contrasts if c.condition == "association" and c.comparator == "off").mean_difference == pytest.approx(-.35)
    with pytest.raises(ValueError, match="种子不匹配"):
        assess(items[:-1], initialization=1701, steps=96)
