"""物理和信息素不变量，不作为学习效果实验。"""
import numpy as np
from mathhackson.colony.geometry import Wall, move_discs
from mathhackson.colony.pheromone import Pheromones


def test_circle_does_not_cross_thin_wall():
    wall=Wall(1,0,0,.12,3)
    p=np.asarray([[-2.,0]],dtype=np.float32)
    q,contact=move_discs(p,np.asarray([[4.,0]],dtype=np.float32),.2,[wall],np.asarray([14.,10.],dtype=np.float32))
    assert q[0,0] <= -.32+1e-4
    assert contact[0]


def test_two_circles_separate():
    p=np.asarray([[0.,0],[.3,0]],dtype=np.float32)
    q,_=move_discs(p,np.zeros_like(p),.2,[],np.asarray([14.,10.],dtype=np.float32))
    assert np.linalg.norm(q[0]-q[1]) >= .4-1e-5


def test_pheromone_does_not_cross_closed_wall_or_wrap():
    f=Pheromones(28,20)
    f.set_walls([Wall(1,0,0,.55,10)])
    f.values[1,10,10]=5
    for _ in range(200): f.tick(.1)
    assert np.all(f.values[1,:,15:] == 0)
    assert np.all(f.values[:,f.blocked] == 0)
    assert np.isfinite(f.values).all()
    g=Pheromones(28,20); g.values[0,0,0]=1; g.tick(.1)
    assert g.values[0,-1,-1]==0
    assert g.values.sum() <= 1
