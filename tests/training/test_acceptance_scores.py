import numpy as np
import pytest

from mathhackson.training.comparison.acceptance_scores import score_world, summarize
from mathhackson.training.foraging.update_curriculum import world_key


def world(seed, index, selected=1., *, count=1, always=0., deliveries=1.):
    result = score_world(world_key(index, seed), seed, (0., .75)[index],
                         np.asarray([selected]), np.ones(1), np.asarray([deliveries]), np.asarray([deliveries]))
    return result.model_copy(update={"count": count, "always_reward": always})


def test_world_scores_include_rejected_points_and_exact_zero_is_rejected():
    result = score_world("test", 1, 0., np.array([4., -2., 10.]), np.array([1., -1., 0.]),
                         np.array([1., -1., 2.]), np.array([2., -1., 3.]))
    assert result.count == 3 and result.accepted == 1
    assert result.selected_reward == pytest.approx(4 / 3)
    assert result.always_reward == 4.
    assert result.selected_focal_deliveries == pytest.approx(1 / 3)
    assert result.positive == 2 and result.negative == 1


def test_conditions_and_seeds_are_equally_weighted_not_pooled_by_count():
    worlds = [world(seed, i, selected=(1. if i == 0 else 3.), count=(1 if i == 0 else 100))
              for seed in (1, 2, 3, 4) for i in (0, 1)]
    result = summarize(worlds, (1, 2, 3, 4), (0., .75), minimum_passing=3)
    assert result.overall_selected == 2. and result.passing_seeds == 4
    assert result.development_continue


def test_three_positive_seeds_do_not_override_negative_overall_or_missing_delivery():
    worlds = [world(seed, i, selected=-10. if seed == 4 else 1.) for seed in (1, 2, 3, 4) for i in (0, 1)]
    result = summarize(worlds, (1, 2, 3, 4), (0., .75), minimum_passing=3)
    assert result.passing_seeds == 3 and result.overall_selected < 0. and not result.development_continue
    worlds = [world(seed, i, deliveries=0.) for seed in (1, 2, 3, 4) for i in (0, 1)]
    assert not summarize(worlds, (1, 2, 3, 4), (0., .75), minimum_passing=3).development_continue


def test_missing_condition_is_not_filled_with_zero_and_duplicate_is_rejected():
    worlds = [world(seed, i) for seed in (1, 2, 3, 4) for i in (0, 1)]
    worlds[-1] = worlds[-1].model_copy(update={"count": 0, "selected_reward": None, "always_reward": None})
    result = summarize(worlds, (1, 2, 3, 4), (0., .75), minimum_passing=3)
    assert result.complete_seeds == 3 and result.overall_selected is None and not result.development_continue
    with pytest.raises(ValueError):
        summarize(worlds + [worlds[0]], (1, 2, 3, 4), (0., .75), minimum_passing=3)


def test_empty_world_and_nonfinite_predictions():
    empty = np.array([])
    result = score_world("empty", 1, 0., empty, empty, empty, empty)
    assert result.count == 0 and result.selected_reward is None
    with pytest.raises(ValueError):
        score_world("bad", 1, 0., np.zeros(1), np.array([float("nan")]), np.zeros(1), np.zeros(1))
