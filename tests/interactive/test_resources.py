import copy

import numpy as np
import pytest

from mathhackson.interactive.world import EditableColony
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.revival import RevivingColony


def idle(env):
    return [DirectionAction(False, 0., 0.) for _ in env.ants]


def conserved(env):
    assert sum(f.stock for f in env.foods) == env.stock
    assert env.stock + sum(a.carrying + a.deliveries for a in env.ants) == env.supplied_stock


def test_multisource_pickup_delivery_and_exhaustion_return_to_original_food():
    env = EditableColony(1, ColonyConfig(ants=1, stock=1, horizon=40))
    food = env.add_food(5., 4., 2)
    ant = env.ants[0]
    ant.position[:] = (5., 4.)
    assert env.step(idle(env))[0].picked_up
    assert food.stock == 1 and env.foods[0].stock == 1 and env.food_origins == [food.id]
    conserved(env)
    ant.position[:] = env.home
    assert env.step(idle(env))[0].delivered
    assert env.food_origins == [None]
    conserved(env)
    ant.position[:] = (5., 4.)
    assert env.step(idle(env))[0].picked_up
    assert not any(s.x == food.x and s.y == food.y for s in env.food_sources())
    ant.exploration_left, ant.reserve_left = 0, 1
    event = env.step(idle(env))[0]
    assert event.exhausted and not event.delivered and event.reward == -2.
    assert food.stock == 1 and env.foods[0].stock == 1 and env.food_origins == [None]
    assert env.release_waiting() == [0]
    conserved(env)


def test_last_piece_is_not_duplicated_across_ants():
    env = EditableColony(1, ColonyConfig(ants=2))
    food = env.add_food(5., 4., 1)
    for ant, x in zip(env.ants, (4.8, 5.2), strict=True):
        ant.position[:] = (x, 4.)
    assert sum(e.picked_up for e in env.step(idle(env))) == 1
    assert food.stock == 0
    conserved(env)


@pytest.mark.parametrize("position,stock", [((0., 0.), 2), ((14., 0.), 2), ((float("nan"), 0.), 2), ((3., 4.), 0), ((3., 4.), True)])
def test_invalid_resource_does_not_mutate_world(position, stock):
    env = EditableColony(1, ColonyConfig())
    before = copy.deepcopy(env.foods)
    with pytest.raises(ValueError):
        env.add_food(*position, stock)
    assert env.foods == before and env.stock == env.supplied_stock


def test_resources_and_signals_are_world_local():
    one, two = [EditableColony(1, ColonyConfig()) for _ in range(2)]
    points = np.asarray([[5., 4.]], dtype=np.float32)
    before = two.signals.sample(points, two.sources()).copy()
    one.add_food(5., 4., 2)
    assert one.signals.sample(points, one.sources())[0, 0] == 4.
    np.testing.assert_array_equal(before, two.signals.sample(points, two.sources()))
    one.signals.trails.values.fill(5.)
    assert not np.array_equal(one.signals.trails.values, two.signals.trails.values)


def test_empty_interactive_world_accepts_more_food_without_reset():
    env = EditableColony(1, ColonyConfig(ants=1, stock=1, horizon=4))
    env.ants[0].position[:] = env.food
    env.step(idle(env))
    env.ants[0].position[:] = env.home
    env.step(idle(env))
    assert env.stock == 0 and not env.done
    env.add_food(5., 4., 2)
    assert env.steps == 2 and env.ants[0].deliveries == 1
    env.step(idle(env))
    env.step(idle(env))
    assert not env.done and env.steps == 4
    conserved(env)


def test_unedited_world_matches_existing_revival_step_for_step():
    config = ColonyConfig(ants=8, stock=16, horizon=80, nest_signal_strength=1., trail_profile="bounded-local-v2")
    old = RevivingColony(1, config, DisturbanceConfig(signal_strength=0., injury_per_step=0.))
    new = EditableColony(1, config)
    for tick in range(80):
        assert old.release_waiting() == new.release_waiting()
        actions = [DirectionAction(tick % 3 != 0, (i - 4) / 8, .5) for i in range(8)]
        assert old.step(actions) == new.step(actions)
        for i in range(8):
            np.testing.assert_array_equal(old.ants[i].position, new.ants[i].position)
            np.testing.assert_array_equal(old.observation(i).vector(), new.observation(i).vector())
        assert old.stock == new.stock
        conserved(new)
