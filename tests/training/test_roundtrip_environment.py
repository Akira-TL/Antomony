import math

import numpy as np
import pytest

from mathhackson.training.roundtrip_environment import RoundTripEnvironment


def test_two_trip_course_picks_up_delivers_and_keeps_two_fields_separate():
    world = RoundTripEnvironment(3)
    world.food = np.asarray([3., 0.], np.float32)
    while not world.carrying:
        world.step(True, 0., True, False)
    assert world.pickups == 1
    assert world.observation()[0] < 0.
    assert world.field.values[0].max() > 0
    assert world.field.values[1].max() == 0
    world.heading = math.pi
    while world.delivered == 0:
        world.step(True, 0., False, True)
    assert world.field.values[1].max() > 0
    assert world.delivered == 1
    assert not world.carrying
    world.heading = 0.
    assert world.observation()[0] > 0.
    assert world.steps < world.horizon
    while not world.carrying:
        world.step(True, 0., True, False)
    world.heading = math.pi
    while not world.done:
        world.step(True, 0., False, True)
    assert world.delivered == 2
    with pytest.raises(ValueError, match="已结束"):
        world.step(True, 0., False, False)


def test_homing_observation_never_contains_nest_direction():
    world = RoundTripEnvironment(4)
    world.position = np.asarray([3., 1.], np.float32)
    world.carrying = True
    before = world.observation()
    world.home = np.asarray([-3., -1.], np.float32)
    assert np.array_equal(world.observation(), before)
    assert before.shape == (17,)
    assert before[0] == before[1] == 0.
    assert before[2] == 0.


def test_private_goal_progress_and_reward_never_enter_observation():
    world = RoundTripEnvironment(4)
    world.carrying = True
    world.progress = .18
    world.reward = 8.
    observation = world.observation()
    assert observation[5] == 0.
    assert observation[7] == 0.


def test_inactive_pheromone_channel_is_hidden_from_each_return_leg():
    world = RoundTripEnvironment(4)
    world.position = np.asarray([1., 0.], np.float32)
    world.field.deposit(world.position, 0, 2.)
    world.field.deposit(world.position, 1, 2.)
    world.carrying = True
    assert np.array_equal(world.observation()[11:14], np.zeros(3))
    world.carrying = False
    world.delivered = 1
    assert np.array_equal(world.observation()[8:11], np.zeros(3))


def test_delivery_uses_nest_area_not_an_exact_point():
    world = RoundTripEnvironment(4)
    world.carrying = True
    world.position = np.asarray([1.3, 0.], np.float32)
    world.heading = math.pi
    world.step(True, 0., False, True)
    assert world.delivered == 1


def test_second_departure_uses_food_scent_outside_visibility():
    world = RoundTripEnvironment(5)
    world.food = np.asarray([4., 0.], np.float32)
    world.delivered = 1
    assert world.observation()[0] == 0.
    world.field.deposit(np.asarray([.5, 0.], np.float32), 1, 1.)
    assert world.observation()[0] > 0.
    assert world.observation()[16] == 0.


def test_stationary_release_requests_do_not_build_a_scent_peak():
    world = RoundTripEnvironment(6)
    for _ in range(12):
        world.step(False, 0., True, True)
    assert world.field.values.max() == 0.
