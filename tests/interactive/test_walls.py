import math

import numpy as np
import pytest

from mathhackson.colony.geometry import Wall
from mathhackson.interactive.signals import BarrierSignals, visible_from
from mathhackson.interactive.world import EditableColony
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.signals import SignalSource, receptor_points


@pytest.mark.parametrize("angle", [0., math.pi / 4, math.pi / 2])
def test_visibility_uses_rotated_rectangle_and_finite_segments(angle):
    wall = Wall(1, 2., 2., .25, 2., angle)
    origin = wall.global_point(np.asarray([-1., 0.]))
    points = np.asarray([wall.global_point(np.asarray(p)) for p in [(1., 0.), (-.5, 0.), (-1., 3.)]])
    assert visible_from(origin, points, [wall]).tolist() == [False, True, True]


def test_deposition_cannot_jump_wall_but_keeps_existing_far_side_signal():
    signals = BarrierSignals("bounded-local-v2")
    wall = Wall(1, 2., 0., .2, 4.)
    signals.trails.set_walls([wall])
    signals.trails.deposit(np.asarray([1.4, 0.]), 0, 1.)
    far = signals.trails.x > 2.2
    assert not signals.trails.values[0, :, far].any()
    assert signals.trails.values[0].any()
    signals.trails.deposit(np.asarray([2.6, 0.]), 0, .5)
    previous = signals.trails.values[0, :, far].copy()
    signals.trails.deposit(np.asarray([1.4, 0.]), 0, 2.)
    np.testing.assert_array_equal(previous, signals.trails.values[0, :, far])


def test_food_smell_and_antennas_do_not_sample_through_wall():
    signals = BarrierSignals("bounded-local-v2")
    points = receptor_points(np.asarray([1.4, 0.], dtype=np.float32), 0.)
    source = SignalSource(2.6, 0., 4., 2., (1., 0., 0., 0., 0., 0., 0., 0.))
    assert signals.sample(points, (source,))[:, 0].any()
    signals.trails.set_walls([Wall(1, 2., 0., .2, 4.)])
    assert not signals.sample(points, (source,)).any()


def test_physics_wall_removal_and_signal_mask_agree():
    env = EditableColony(1, ColonyConfig(ants=1, horizon=80, exploration_steps=200))
    wall = env.add_wall(3., 2., .25, 2.)
    env.ants[0].position[:] = (2.4, 2.)
    env.ants[0].heading = 0.
    action = [DirectionAction(True, 0., 1.)]
    for _ in range(20):
        env.step(action)
    assert env.ants[0].position[0] <= 2.571 and env.ants[0].contact
    assert env.signals.trails.blocked.any()
    env.remove_wall(wall.id)
    assert not env.signals.trails.blocked.any()
    for _ in range(20):
        env.step(action)
    assert env.ants[0].position[0] > 3.25


def test_wall_rejects_ants_resources_nest_and_does_not_relocate():
    env = EditableColony(1, ColonyConfig(ants=1))
    env.ants[0].position[:] = (5., 4.)
    before = env.ants[0].position.copy()
    for x, y in [(0., 0.), (5., 4.), tuple(env.food)]:
        with pytest.raises(ValueError):
            env.add_wall(x, y, .25, 1.)
    assert not env.walls and not env.signals.trails.blocked.any()
    np.testing.assert_array_equal(before, env.ants[0].position)
    env.add_wall(3., 2., .25, 2.)
    with pytest.raises(ValueError):
        env.add_food(3., 2., 1)
