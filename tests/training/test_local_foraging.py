import math

import numpy as np
import pytest

from mathhackson.training.foraging.environment import ForagingConfig, ForagingEnvironment
from mathhackson.training.foraging.signals import LocalSignals, SignalSource


def test_finite_food_signal_has_no_response_outside_radius():
    source = SignalSource(0., 0., 2., 4., (1., 0., 0., 0., 0., 0., 0., 0.))
    values = source.sample(np.asarray([[0., 0.], [1., 0.], [2., 0.], [8., 0.]]))
    np.testing.assert_array_equal(values[:, 0], [4., 1., 0., 0.])
    assert not values[:, 1:].any()


def test_combination_sources_add_responses_without_action_labels():
    field = LocalSignals()
    sources = (SignalSource(0., 0., 2., 2., (1., 0., 0., .5, 0., 0., 0., .25)),
               SignalSource(0., 0., 2., 1., (1., 0., 0., 0., 0., 0., 0., 0.)))
    np.testing.assert_array_equal(field.sample(np.asarray([[0., 0.]]), sources), [[3., 0., 0., 1., 0., 0., 0., .5]])


def test_distant_food_coordinates_do_not_change_observation():
    first, second = ForagingEnvironment(12), ForagingEnvironment(12)
    first.food = np.asarray([8., 0.], dtype=np.float32)
    second.food = np.asarray([-8., 0.], dtype=np.float32)
    np.testing.assert_array_equal(first.observation().vector(), second.observation().vector())
    assert first.observation().vector().shape == (76,)
    assert not first.observation().receptors[:, 3:].any()


def test_stationary_empty_ant_deposits_and_reward_decreases_without_movement_condition():
    env = ForagingEnvironment(12)
    env.food = np.asarray([8., 0.], dtype=np.float32)
    position = env.position.copy()
    first = env.step(False, 0.)
    for _ in range(15):
        last = env.step(False, 0.)
    np.testing.assert_array_equal(env.position, position)
    assert env.signals.trails.values[0].sum() > 0.
    assert not env.signals.trails.values[1].any()
    assert 0. < last.exploration_reward < first.exploration_reward


def test_pickup_and_delivery_reward_once_and_switch_fixed_release():
    env = ForagingEnvironment(13)
    env.position = env.food.copy()
    first = env.step(False, 0.)
    assert first.picked_up and first.reward == 1.
    assert not env.step(False, 0.).picked_up
    assert env.pickups == 1 and env.stock == 1
    home_field = env.signals.trails.values[0].copy()
    env.step(True, 0.)
    env.step(True, 0.)
    assert env.signals.trails.values[1].sum() > 0.
    assert env.signals.trails.values[0].sum() <= home_field.sum()
    env.position = env.home.copy()
    delivery = env.step(False, 0.)
    assert delivery.delivered and delivery.reward == 4.
    assert not env.step(False, 0.).delivered
    assert env.deliveries == 1


def test_bilinear_trail_sampling_preserves_linear_field_and_strength():
    field = LocalSignals()
    grid = field.trails
    x = -14. + (np.arange(grid.width) + .5) * 28. / grid.width
    grid.values[0] = 2. + .1 * x
    values = field.trail_samples(np.asarray([[0., 0.], [1.25, 2.]]))
    np.testing.assert_allclose(values[:, 0], [2., 2.125], atol=1e-6)


def test_individual_environments_do_not_share_pheromones():
    first, second = ForagingEnvironment(8), ForagingEnvironment(8)
    first.step(False, 0.)
    assert first.signals.trails.values.sum() > 0.
    assert second.signals.trails.values.sum() == 0.


def test_world_rejects_bad_action_and_terminal_step():
    env = ForagingEnvironment(2, ForagingConfig(horizon=1))
    with pytest.raises(ValueError):
        env.step(True, math.nan)
    env.step(True, 0.)
    with pytest.raises(ValueError):
        env.step(True, 0.)
