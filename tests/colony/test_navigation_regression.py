"""用户反馈的方向反复、虚假气味聚集与动态放墙回归。"""
import math
import numpy as np
from mathhackson.colony.world import World


def empty_world(seed=37):
    w=World(seed,1,warmup=100)
    w.walls=[]
    w.foods=[]
    w.field.set_walls([])
    a=w.ants[0]
    a.position=np.zeros(2,np.float32)
    a.heading=a.wander=0.
    a.home=w.nest-a.position
    return w


def test_heading_does_not_flip_between_physics_steps():
    w=empty_world()
    w.ants[0].wander=math.pi
    changes=[]
    for _ in range(20):
        old=w.ants[0].heading
        w.tick()
        changes.append(abs(math.atan2(math.sin(w.ants[0].heading-old),math.cos(w.ants[0].heading-old))))
    assert max(changes)<=.46, changes


def test_empty_signal_does_not_hold_an_ant_at_peak_forever():
    w=empty_world(41)
    w.field.spray(np.asarray([1.,0.],np.float32))
    distances=[]
    for _ in range(240):
        w.tick()
        distances.append(float(np.linalg.norm(w.ants[0].position-np.asarray([1.,0.]))))
    assert max(distances[-80:])>3.0, distances[-10:]


def test_home_channel_does_not_attract_empty_ants():
    a,b=empty_world(12),empty_world(12)
    b.field.values[0].fill(8.)
    np.testing.assert_array_equal(a.direction(a.ants[0]),b.direction(b.ants[0]))
