"""方向投影比较的独立算术判据；只用合成汇总值。"""
from __future__ import annotations

import pytest

from mathhackson.training.foraging.feedback_cues import CONDITIONS
from mathhackson.training.foraging.plastic_course.assessment import SeedAssessment
from mathhackson.training.foraging.plastic_course.run import EvaluationRow
from scripts.analyses.plastic_projection import AcrossContrast, contrast, decide


def row(seed: int, values: list[float], last: list[float]) -> EvaluationRow:
    return EvaluationRow(initialization=1711, condition="association", batch_seed=seed, mode="learned",
        mean_after_feedback=values, mean_last_quarter=last, acceptances=[1] * len(values),
        nonzero_writes=[1] * len(values), total_variation_from_initial=[0.] * len(values))


def eligible() -> SeedAssessment:
    return SeedAssessment(initialization=1711, contrasts=[], accepted_fraction=.5, nonzero_writes=1,
        association_gain=True, reference_preserved=True, learned_timing_gain=True, mixed_control=True, passed=True)


def differences() -> list[AcrossContrast]:
    return [AcrossContrast(initialization=1711, condition=condition, comparison=comparison,
        differences=[.03], last_quarter_differences=[.03],
        mean_difference=.03 if condition == "association" else -.02,
        last_quarter_difference=.03 if condition == "association" else -.02)
        for condition in CONDITIONS for comparison in ("architecture", "training")]


def test_paired_contrast_preserves_negative_episodes_and_equal_weight() -> None:
    result = contrast([row(2, [.3, .1], [.2, -.2]), row(1, [.1, -.2], [.1, 0.])],
        [row(1, [0., -.1], [0., .1]), row(2, [.2, .2], [.1, -.1])],
        initialization=1711, condition="association", comparison="training")
    assert result.differences == pytest.approx([.1, -.1, .1, -.1])
    assert result.last_quarter_differences == pytest.approx([.1, -.1, .1, -.1])
    assert result.mean_difference == pytest.approx(0.)
    assert result.last_quarter_difference == pytest.approx(0.)


def test_pairing_rejects_distinct_batch_identity() -> None:
    with pytest.raises(ValueError, match="配对身份不匹配"):
        contrast([row(1, [.2], [.2])], [row(2, [.1], [.1])],
            initialization=1711, condition="association", comparison="architecture")


def test_all_original_and_added_conditions_required_at_inclusive_thresholds() -> None:
    assert decide(1711, eligible(), differences()).passed
    assert not decide(1711, eligible().model_copy(update={"passed": False}), differences()).passed


@pytest.mark.parametrize("comparison", ["architecture", "training"])
@pytest.mark.parametrize("condition", ["association", "reference"])
@pytest.mark.parametrize("field", ["mean_difference", "last_quarter_difference"])
def test_any_added_condition_failure_prevents_adoption(comparison: str, condition: str, field: str) -> None:
    values = differences()
    target = next(item for item in values if item.comparison == comparison and item.condition == condition)
    setattr(target, field, getattr(target, field) - .0001)
    assert not decide(1711, eligible(), values).passed


def test_missing_condition_cannot_be_interpreted_as_pass() -> None:
    with pytest.raises(ValueError, match="缺少完整条件"):
        decide(1711, eligible(), differences()[:-1])
