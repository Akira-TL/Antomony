"""可观察的循迹行为回归；不是学习收益结论。"""
import math
import numpy as np
from mathhackson.colony.world import World
from mathhackson.colony.geometry import unit


def setup_trail_world():
    world = World(seed=77, count=1, warmup=20)
    world.walls = []; world.foods = []; world.field.set_walls([])
    ant = world.ants[0]
    ant.position = np.zeros(2, np.float32)
    ant.heading = 0.; ant.last_mark_position = ant.position.copy()
    ant.waypoint = np.asarray([2., 0.], np.float32)
    ant.waypoint_deadline = 100
    return world, ant


def test_food_trail_can_override_an_unrelated_exploration_heading():
    world, ant = setup_trail_world()
    aim = unit(math.pi / 3)
    for distance in np.linspace(.3, 2., 30):
        world.field.deposit(aim * distance, 1, .35)
    world.sense(ant, np.stack([ant.position]))
    wanted = world.direction(ant)
    alignment = float(wanted @ aim / np.linalg.norm(wanted))
    assert alignment > .93, f'食物气味必须能改变探索朝向，实际对齐度 {alignment:.3f}'
