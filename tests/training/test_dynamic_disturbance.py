import copy
import math

import numpy as np
import pytest
from pydantic import ValidationError

from mathhackson.training.direction.environment import MAX_TURN, STEP_DISTANCE
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony


STAY = [DirectionAction(False, 0., 0.)]
FORWARD = [DirectionAction(True, 0., 1.)]


def world(**kwargs):
    return DisturbedColony(42, ColonyConfig(ants=1, horizon=40),
                           DisturbanceConfig(injury_per_step=0., **kwargs))


def test_slowdown_changes_translation_not_turn_or_budget_cost():
    env = world(speed_multiplier=.25)
    ant = env.ants[0]
    ant.position = env.source_position.copy()
    ant.heading = 0.
    initial = ant.position.copy()
    budget = ant.exploration_left
    env.step([DirectionAction(True, 1., 1.)])
    np.testing.assert_allclose(ant.position - initial,
                               STEP_DISTANCE * .25 * np.array([math.cos(MAX_TURN), math.sin(MAX_TURN)]), atol=1e-7)
    assert ant.heading == pytest.approx(MAX_TURN)
    assert ant.exploration_left == budget - 1
    assert ant.previous_move and ant.previous_turn == 1.


def test_default_disturbance_retains_base_motion_exactly():
    base = ColonyEnvironment(42, ColonyConfig(ants=1, horizon=40))
    env = world(signal_strength=0.)
    for _ in range(10):
        assert base.step(FORWARD) == env.step(FORWARD)
        np.testing.assert_array_equal(base.ants[0].position, env.ants[0].position)
        np.testing.assert_array_equal(base.observation(0).vector(), env.observation(0).vector())


def test_slowdown_has_local_support_and_does_not_move_stopped_ant():
    env = world(speed_multiplier=.25, slowdown_radius=.5)
    ant = env.ants[0]
    ant.position = env.source_position + np.array([1., 0.], dtype=np.float32)
    ant.heading = 0.
    initial = ant.position.copy()
    env.step(FORWARD)
    assert ant.position[0] - initial[0] == pytest.approx(STEP_DISTANCE, abs=1e-7)
    ant.position = env.source_position.copy()
    initial = ant.position.copy()
    env.step(STAY)
    np.testing.assert_array_equal(ant.position, initial)


def test_periodic_schedule_boundaries_and_local_signal():
    env = world(active_from=2, active_until=8, period_steps=4, active_steps=2,
                inactive_signal_scale=0., speed_multiplier=.25)
    ant = env.ants[0]
    ant.position = env.source_position.copy()
    expected = [False, False, True, True, False, False, True, True, False, False]
    for tick, active in enumerate(expected):
        assert env.steps == tick
        assert env.disturbance.active_at(tick) is active
        assert bool(env.observation(0).receptors[:, 3].any()) is active
        assert env.movement_scale(0) == (.25 if active else 1.)
        env.step(STAY)


def test_periodic_injury_uses_same_action_tick_as_schedule():
    env = DisturbedColony(42, ColonyConfig(ants=1), DisturbanceConfig(
        injury_per_step=.1, active_from=2, period_steps=4, active_steps=2))
    env.ants[0].position = env.source_position.copy()
    for expected in (0., 0., .1, .2, .2, .2, .3, .4):
        env.step(STAY)
        assert env.injuries[0] == pytest.approx(expected)


def test_prescribed_motion_is_synchronous_despite_different_actions_and_isolated_fields():
    first = world(motion_amplitude=1., motion_period_steps=8)
    second = world(motion_amplitude=1., motion_period_steps=8)
    first.signals.trails.deposit(np.zeros(2, dtype=np.float32), 0, 1.)
    assert not np.array_equal(first.signals.trails.values, second.signals.trails.values)
    assert not np.shares_memory(first.signals.trails.values, second.signals.trails.values)
    for tick in range(9):
        expected = first.source_center + first.motion_axis * math.sin(2. * math.pi * tick / 8)
        np.testing.assert_allclose(first.source_position, expected, atol=1e-7)
        np.testing.assert_array_equal(first.source_position, second.source_position)
        np.testing.assert_array_equal([first.extra_sources[0].x, first.extra_sources[0].y], first.source_position)
        first.step(STAY)
        second.step(FORWARD)


def test_moving_source_remains_local_and_observation_has_no_new_labels():
    env = world(motion_amplitude=1., motion_period_steps=8, signal_radius=.5)
    env.step(STAY)
    env.ants[0].position = env.source_position + np.array([3., 0.], dtype=np.float32)
    observation = env.observation(0)
    assert observation.vector().shape == (78,)
    assert not observation.receptors[:, 3:].any()
    clone = copy.deepcopy(env)
    clone.step(STAY)
    assert env.steps == 1 and clone.steps == 2
    np.testing.assert_allclose(env.source_position, env.source_center + env.motion_axis / math.sqrt(2.), atol=1e-7)


@pytest.mark.parametrize("values", [
    {"period_steps": 8}, {"active_steps": 2}, {"period_steps": 4, "active_steps": 5},
    {"active_from": 3, "active_until": 3}, {"speed_multiplier": 0.},
    {"response": (-1., 0., 0., 0., 0., 0., 0., 0.)},
])
def test_invalid_schedules_are_rejected(values):
    with pytest.raises(ValidationError):
        DisturbanceConfig(**values)


def test_motion_cannot_leave_world_bounds():
    with pytest.raises(ValueError, match="超出场地"):
        world(motion_amplitude=1000.)
