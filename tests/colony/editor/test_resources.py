"""资源编辑和展示的生命周期，气味仍按环境过程衰减。"""
import numpy as np
from mathhackson.colony.world import World, Food
from mathhackson.colony.protocol import snapshot


def test_depleted_resource_marker_disappears_without_erasing_remote_scent():
    world = World(13, 1, warmup=5)
    world.foods = [Food(1, 0., 0., amount=1)]
    ant = world.ants[0]; ant.position = np.zeros(2, np.float32)
    world.field.deposit(np.asarray([1., 0.], np.float32), 1, 1.)
    world.collect(ant)
    assert ant.carrying
    assert snapshot(world).foods == []
    assert world.field.values[1].sum() == 1.
    assert '已放置' in world.add_wall(0., 0., .3, 1.)


def test_resource_preview_does_not_add_food_and_matches_rejection():
    world = World(23, 1, warmup=5)
    before = [(f.id, f.amount) for f in world.foods]
    valid, _ = world.plan_food(-10., 0.)
    assert not valid
    valid, message = world.plan_food(9., 7.)
    assert valid, message
    assert [(f.id, f.amount) for f in world.foods] == before
    assert '已放置' in world.add_food(9., 7.)
    assert not world.plan_food(9., 7.)[0]


def test_new_food_ids_do_not_reuse_ids_after_depletion():
    world = World(23, 1, warmup=5)
    for food in world.foods: food.amount = 0
    assert '已放置' in world.add_food(9., 7.)
    assert world.foods[-1].id == 4
