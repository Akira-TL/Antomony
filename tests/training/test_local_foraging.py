import math

import numpy as np
import pytest
import torch

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


def test_unused_receptor_connections_stay_zero_and_frozen_during_basic_training():
    from mathhackson.training.foraging.policy import ForagingPolicy
    from mathhackson.training.foraging.curriculum import signal_batch
    policy = ForagingPolicy(71)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=.003)
    inputs, targets = signal_batch(np.random.default_rng(71), 32)
    assert not inputs[:, :72].reshape(32, 9, 8)[:, :, 3:].any()
    for _ in range(3):
        output, _, _ = policy(inputs)
        loss = 1. - (output * targets).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert not policy.novel_signal.weight.requires_grad
    assert policy.novel_signal.weight.grad is None
    assert not policy.novel_signal.weight.any()
    changed = inputs.clone()
    changed[:, :72].reshape(32, 9, 8)[:, :, 3:] = 1.
    torch.testing.assert_close(policy(inputs)[0], policy(changed)[0], rtol=0., atol=0.)


def test_local_policy_snapshot_roundtrip_and_individual_motor_freeze(tmp_path):
    from mathhackson.training.foraging.policy import ForagingController, ForagingPolicy
    from mathhackson.training.direction.policy import DirectionMotor
    policy = ForagingPolicy(8)
    path = tmp_path / "policy.npz"
    policy.save(path, update=0, phase="signal")
    restored = ForagingPolicy.load(path)
    observation = ForagingEnvironment(4).observation()
    torch.testing.assert_close(policy(torch.from_numpy(observation.vector()))[0],
                               restored(torch.from_numpy(observation.vector()))[0], rtol=0., atol=0.)
    motor = DirectionMotor(41)
    controller = ForagingController(restored, motor)
    before = [p.detach().clone() for p in motor.parameters()]
    decision = controller.decide(observation)
    assert np.linalg.norm(decision.direction) == pytest.approx(1.)
    assert all(not p.requires_grad for p in motor.parameters())
    assert all(a.equal(b) for a, b in zip(before, motor.parameters(), strict=True))


def test_recurrent_direction_uses_recent_and_sparse_hidden_frames():
    from mathhackson.training.foraging.policy import ForagingPolicy
    policy = ForagingPolicy(8)
    with torch.no_grad():
        policy.observation_layer.weight.zero_()
        policy.observation_layer.bias.zero_()
        policy.recent_memory.weight.fill_(.1)
        policy.sparse_memory.weight.fill_(.1)
    observation = torch.zeros(76)
    history = [torch.zeros(8) for _ in range(16)]
    history[-5] = torch.ones(8)
    assert not policy(observation, tuple(history))[1].any()
    history[-8] = torch.ones(8)
    torch.testing.assert_close(policy(observation, tuple(history))[1], torch.full((8,), .66403677))
    history[-8] = torch.zeros(8)
    history[-4] = torch.ones(8)
    torch.testing.assert_close(policy(observation, tuple(history))[1], torch.full((8,), .92166855))
