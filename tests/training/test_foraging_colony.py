import numpy as np
import pytest

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment


def idle(env):
    return [DirectionAction(False, 0., 0.) for _ in env.ants]


def test_shared_field_ticks_once_and_worlds_stay_isolated(monkeypatch):
    env, other = ColonyEnvironment(8), ColonyEnvironment(8)
    ticks = []
    original = env.signals.trails.tick
    monkeypatch.setattr(env.signals.trails, "tick", lambda dt: (ticks.append(dt), original(dt)))
    env.step(idle(env))
    assert ticks == [.1]
    assert all(env.observation(i).receptors[:, 1].any() for i in range(8))
    assert not other.signals.trails.values.any()
    assert env.observation(0).vector().shape == (78,)
    assert not env.observation(0).receptors[:, 3:].any()


def test_zero_exploration_budget_preserves_movement_without_forced_turn():
    env = ColonyEnvironment(8, ColonyConfig(ants=1, exploration_steps=1, reserve_steps=2))
    ant = env.ants[0]
    ant.position = np.array([3., 0.], dtype=np.float32)
    ant.heading = 0.
    first = env.step([DirectionAction(True, 0., 1.)])[0]
    assert not first.exhausted and first.exploration_reward == 0.
    assert ant.exploration_left == 0 and ant.reserve_left == 2
    x = ant.position[0]
    env.step([DirectionAction(True, 0., 1.)])
    assert ant.position[0] > x and ant.heading == 0.
    last = env.step(idle(env))[0]
    assert last.exhausted and last.reward == -2. and env.done


def test_budget_return_refills_once_without_food_and_resets_release_origin():
    env = ColonyEnvironment(8, ColonyConfig(ants=1))
    ant = env.ants[0]
    ant.position = env.home.copy()
    ant.away = True
    ant.exploration_left = 0
    ant.reserve_left = 1
    ant.leg_distance = 8.
    outcome = env.step(idle(env))[0]
    assert outcome.budget_return and outcome.reward == 2. and not outcome.exhausted
    assert ant.exploration_left == 160 and ant.reserve_left == 160 and ant.leg_distance == 0.
    assert not env.step(idle(env))[0].budget_return
    assert ant.budget_returns == 1 and ant.deliveries == 0


def test_waiting_inside_home_does_not_farm_refill_reward():
    env = ColonyEnvironment(8, ColonyConfig(ants=1, exploration_steps=1, reserve_steps=1))
    assert not env.step(idle(env))[0].budget_return
    outcome = env.step(idle(env))[0]
    assert not outcome.budget_return and outcome.exhausted


def test_stock_cannot_be_picked_up_twice_in_same_tick():
    env = ColonyEnvironment(8, ColonyConfig(ants=2, stock=1))
    for ant, offset in zip(env.ants, [-.2, .2], strict=True):
        ant.position = env.food + np.array([offset, 0.], dtype=np.float32)
    events = env.step(idle(env))
    assert sum(event.picked_up for event in events) == 1
    assert env.stock == 0
    assert sum(ant.pickups for ant in env.ants) == 1


def test_invalid_actions_do_not_partially_advance_world():
    env = ColonyEnvironment(8)
    before = [ant.heading for ant in env.ants]
    with pytest.raises(ValueError):
        env.step([DirectionAction(True, float("nan"), 1.)] * 8)
    assert env.steps == 0 and before == [ant.heading for ant in env.ants]
