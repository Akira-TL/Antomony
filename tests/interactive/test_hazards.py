import numpy as np
import pytest

from mathhackson.interactive.hazards import Trap
from mathhackson.interactive.world import EditableColony
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig


def environment():
    return EditableColony(1, ColonyConfig(ants=1, stock=1, horizon=80, exploration_reward=0.))


def test_trap_hurts_before_killing_and_returns_food_to_correct_source():
    env = environment()
    food = env.add_food(5., 4., 1)
    env.ants[0].position[:] = (5., 4.)
    action = [DirectionAction(False, 0., 0.)]
    env.step(action)
    env.add_trap(Trap(id=0, x=5., y=4., born_at=1, injury=.2))
    event = env.step(action)[0]
    assert not event.exhausted and event.reward == -.2
    assert env.injuries[0] == pytest.approx(.2)
    assert env.observation(0).receptors[:, 3].max() > 0.
    for _ in range(3):
        assert not env.step(action)[0].exhausted
    assert env.step(action)[0].exhausted
    assert env.deaths[0] == 1 and env.terminations[0] == 1
    assert env.ants[0].deliveries == 0 and food.stock == 1 and env.stock == 2
    assert env.total_injury[0] == pytest.approx(1.)
    assert env.release_waiting() == [0]
    assert env.revivals[0] == 1 and env.injuries[0] == 0


def test_period_uses_physics_ticks_and_does_not_hide_source_completely():
    env = environment()
    env.ants[0].position[:] = (5., 4.)
    trap = Trap(id=0, x=5., y=4., born_at=0, injury=.1, period=4)
    env.add_trap(trap)
    rewards = [env.step([DirectionAction(False, 0., 0.)])[0].reward for _ in range(4)]
    assert rewards == [-.1, -.1, 0., 0.]
    assert trap.source(2).strength == .5 and trap.source(4).strength == 2.


def test_slowdown_changes_actual_motion_without_forced_direction():
    env, plain = environment(), environment()
    for world in (env, plain):
        world.ants[0].position[:] = (5., 4.)
        world.ants[0].heading = 0.
    env.add_trap(Trap(id=0, x=5., y=4., born_at=0, injury=0., speed_multiplier=.25))
    action = [DirectionAction(True, 0., 1.)]
    env.step(action)
    plain.step(action)
    assert env.ants[0].position[0] - 5. == pytest.approx((plain.ants[0].position[0] - 5.) * .25, abs=1e-6)
    assert env.ants[0].heading == plain.ants[0].heading == 0.
    assert env.injuries[0] == 0.


def test_moving_source_and_effect_share_same_position_at_action_tick():
    env = environment()
    trap = Trap(id=0, x=5., y=4., born_at=0, motion_amplitude=1., motion_period=8)
    env.add_trap(trap)
    env.steps = 2
    env.ants[0].position[:] = (5., 5.)
    np.testing.assert_allclose(trap.position(env.steps), (5., 5.))
    assert trap.source(env.steps).y == 5.
    assert env.step([DirectionAction(False, 0., 0.)])[0].reward == -.08


def test_invalid_placement_rejects_whole_change_and_removal_stops_harm():
    env = environment()
    for x, y in ((0., 0.), (14., 0.)):
        with pytest.raises(ValueError):
            env.add_trap(Trap(id=0, x=x, y=y, born_at=0))
    assert not env.traps
    env.ants[0].position[:] = (5., 4.)
    env.add_trap(Trap(id=0, x=5., y=4., born_at=0))
    with pytest.raises(ValueError):
        env.add_trap(Trap(id=1, x=5., y=4., born_at=0))
    env.remove_trap(0)
    assert env.step([DirectionAction(False, 0., 0.)])[0].reward == 0.
