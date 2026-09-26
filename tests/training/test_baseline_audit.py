import pytest

from mathhackson.training.comparison.baseline_audit import Pair, next_baseline, worth_further_validation
from mathhackson.training.comparison.return_reward_audit import Totals


def pair(seed, delivered=0, died=0, exhausted=0):
    return Pair(seed=seed, arm='always', disabled=Totals(), enabled=Totals(),
        delivery_difference=delivered, death_difference=died, exhaustion_difference=exhausted,
        reward_difference=0., write_difference=0)


def test_baseline_reconstruction_matches_literal_example():
    assert next_baseline(0., [2., 4.], .5, .5) == 2.
    assert next_baseline(2., [-2., 0.], .5, .5) == .5
    assert next_baseline(2., [-2.], .5, .5) == 0.
    with pytest.raises(ValueError):
        next_baseline(0., [], .5, .5)


def test_adoption_requires_both_worlds_without_hidden_exhaustion_tradeoff():
    assert worth_further_validation([pair(1, delivered=1), pair(2)])
    assert worth_further_validation([pair(1, died=-1), pair(2)])
    assert not worth_further_validation([pair(1), pair(2)])
    assert not worth_further_validation([pair(1, delivered=1), pair(2, exhausted=1)])
    assert not worth_further_validation([pair(1, died=-1), pair(2, delivered=-1)])
    with pytest.raises(ValueError):
        worth_further_validation([pair(1), pair(1)])
