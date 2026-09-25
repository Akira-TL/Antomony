"""旋转后的实体边界必须与编辑角度一致。"""
import math
import numpy as np
from mathhackson.colony.geometry import Wall, move_discs, ray_distance, unit
from mathhackson.colony.world import World
from mathhackson.colony.pheromone import Pheromones


def test_rotated_wall_blocks_its_local_normal_not_its_old_axis():
    angle = math.pi / 4
    wall = Wall(1, 0., 0., .2, 3., angle=angle)
    normal = unit(angle)
    assert wall.overlaps(unit(angle + math.pi/2)*2., .2)
    assert not wall.overlaps(np.asarray([0., 2.], np.float32), .2)
    distance = ray_distance(-2*normal, normal, [wall], np.asarray([14., 10.], np.float32), .2, reach=4.)
    assert abs(distance-1.6) < 1e-4
    moved, contact = move_discs(np.asarray([-2*normal]), np.asarray([4*normal]), .2, [wall], np.asarray([14., 10.], np.float32))
    assert float(moved[0]@normal) <= -.4+1e-4
    assert contact[0]


def test_rotated_placement_moves_covered_ant_without_changing_its_model():
    world = World(42, 1, warmup=5)
    ant = world.ants[0]
    ant.position = np.zeros(2, np.float32); ant.home = world.nest-ant.position
    before = ant.brain.fingerprint()
    assert '已放置' in world.add_wall(0., 0., .3, 1., angle=math.pi/4)
    assert not world.walls[-1].overlaps(ant.position, world.radius)
    assert ant.brain.fingerprint() == before
    assert ant.brain.updates == 0
    np.testing.assert_allclose(ant.home, world.nest-ant.position)


def test_rotated_field_mask_has_no_open_diagonal_gap():
    field = Pheromones()
    field.set_walls([Wall(1, 0., 0., .3, 20., angle=math.pi/4)])
    field.deposit(np.asarray([-2., -2.], np.float32), 1, 1.)
    for _ in range(180): field.tick(.1)
    assert field.sample(np.asarray([2., 2.], np.float32), 1) == 0.
