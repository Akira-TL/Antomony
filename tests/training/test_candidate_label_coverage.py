import itertools

import pytest

from mathhackson.training.comparison.candidate_label_coverage import Label, signs, summarize


def label(deliveries, reward, individual=0):
    return Label(partition='train', world='test', seed=1, initial_norm=0., tick=1, individual=individual,
        skip_deliveries=2, accept_deliveries=2+deliveries, delivery_difference=deliveries,
        skip_reward=0., accept_reward=reward, reward_difference=reward)


def test_binary_selector_upper_bound_matches_exhaustive_enumeration():
    differences = [-2, 0, 1, 2]
    rows = [label(value, value) for value in differences]
    exact = max(sum(a*d for a, d in zip(choices, differences, strict=True))
                for choices in itertools.product((0, 1), repeat=len(rows)))
    result = summarize(rows, 'train')
    assert result.maximum_delivery_sum == exact == 3
    assert result.maximum_delivery_mean == .75


def test_positive_reward_cannot_create_missing_delivery_improvements():
    result = summarize([label(0, 1), label(-1, 2), label(0, 0)], 'train')
    assert result.rewards.positive == 2 and result.deliveries.positive == 0
    assert result.maximum_delivery_sum == 0
    assert result.always_delivery_sum == -1


def test_missing_groups_remain_missing_and_tolerance_is_only_for_reward():
    result = summarize([label(0, 1e-7)], 'train', individual=1)
    assert result.count == 0 and result.maximum_delivery_mean is None
    assert signs([-1e-7, 0., 1e-7, 2e-6], 1e-6).model_dump() == {'positive': 1, 'negative': 0, 'zero': 3}
    with pytest.raises(ValueError):
        signs([float('nan')], 0.)
