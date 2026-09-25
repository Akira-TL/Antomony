"""返巢绕障时的标记选择回归；不把规则修复计作神经学习收益。"""
from __future__ import annotations

import math

import numpy as np

from mathhackson.colony.geometry import Wall, ray_distance, unit
from mathhackson.colony.world import World


def returning_world() -> World:
    world = World(seed=73, count=1, warmup=0)
    world.walls = [Wall(1, -1.0, 0.0, .25, 1.4)]
    world.foods = []
    world.field.set_walls(world.walls)
    ant = world.ants[0]
    ant.position = np.zeros(2, dtype=np.float32)
    ant.home = world.nest - ant.position
    ant.heading = 0.0
    ant.carrying = True
    ant.last_mark_position = ant.position.copy()
    ant.last_mark_tick = world.tick_count
    ant.waypoint = np.asarray([2.0, 0.0], dtype=np.float32)
    ant.waypoint_deadline = 100
    world.sense(ant, np.stack([ant.position]))
    return world


def test_returner_uses_visible_home_trail_when_direct_return_is_blocked() -> None:
    plain, scented = returning_world(), returning_world()
    marker = 1.1 * np.asarray([math.cos(.9), math.sin(.9)], dtype=np.float32)
    scented.field.deposit(marker, channel=0, amount=2.0)
    ant = scented.ants[0]
    assert scented.field.sample(marker, 0) == 2.0
    assert scented.field.sample(marker, 1) == 0.0
    assert ray_distance(ant.position, unit(.9), scented.walls, scented.half, scented.radius) >= 1.1
    assert ray_distance(ant.position, unit(math.pi), scented.walls, scented.half, scented.radius) < 2.4
    base = plain.direction(plain.ants[0])
    guided = scented.direction(scented.ants[0])
    assert float(guided[1] - base[1]) > .05, (base, guided)


def test_returner_does_not_follow_its_own_food_channel() -> None:
    plain, scented = returning_world(), returning_world()
    marker = 1.1 * np.asarray([math.cos(.9), math.sin(.9)], dtype=np.float32)
    scented.field.deposit(marker, channel=1, amount=2.0)
    np.testing.assert_array_equal(
        plain.direction(plain.ants[0]), scented.direction(scented.ants[0])
    )


def test_clear_direct_return_is_not_diverted_by_unrelated_home_peak() -> None:
    plain, scented = returning_world(), returning_world()
    for world in (plain, scented):
        world.walls = []
        world.field.set_walls([])
    scented.field.deposit(np.asarray([1.1, 0.0], dtype=np.float32), channel=0, amount=8.0)
    np.testing.assert_array_equal(
        plain.direction(plain.ants[0]), scented.direction(scented.ants[0])
    )
