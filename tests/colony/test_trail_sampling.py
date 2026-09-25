"""标记间距与有效局部探索的工程回归，不作为学习增益实验。"""
import math
import numpy as np
from mathhackson.colony.world import World
from mathhackson.colony.navigation import deposit_trail


def world_at_origin():
    world=World(61,1,warmup=40)
    world.walls=[]; world.foods=[]; world.field.set_walls([])
    ant=world.ants[0]
    ant.position=np.zeros(2,np.float32); ant.home=world.nest-ant.position
    ant.heading=0.; ant.last_mark_position=ant.position.copy()
    ant.last_mark_tick=0
    return world,ant


def test_back_and_forth_jitter_does_not_reinforce_one_mark():
    world,ant=world_at_origin()
    for tick in range(60):
        old=ant.position.copy()
        ant.position=np.asarray([.03 if tick%2 else -.03,0.],np.float32)
        world.tick_count=tick
        deposit_trail(world,ant,ant.position-old)
    assert world.field.values.sum()==0


def test_new_mark_waits_until_body_leaves_previous_mark():
    world,ant=world_at_origin()
    ant.position[0]=.10
    deposit_trail(world,ant,np.asarray([.1,0],np.float32))
    assert world.field.values.sum()==0
    ant.position[0]=.40
    deposit_trail(world,ant,np.asarray([.3,0],np.float32))
    assert world.field.values.sum()>0
    before=world.field.values.copy()
    deposit_trail(world,ant,np.zeros(2,np.float32))
    np.testing.assert_array_equal(world.field.values,before)


def test_source_age_and_field_age_are_distinct_decay_mechanisms():
    young,ay=world_at_origin(); old,ao=world_at_origin()
    ay.age=0.; ao.age=30.
    for world,ant in ((young,ay),(old,ao)):
        ant.position[0]=.4
        deposit_trail(world,ant,np.asarray([.4,0],np.float32))
    assert 0<float(old.field.values.sum())<float(young.field.values.sum())
    total=float(young.field.values.sum())
    young.field.tick(.1)
    assert 0<float(young.field.values.sum())<total


def test_exploration_intent_uses_valid_samples_near_corner():
    world,ant=world_at_origin()
    ant.position=np.asarray([13.6,9.6],np.float32)
    ant.heading=math.pi/4
    world.sense(ant,np.stack([ant.position]))
    wanted=world.direction(ant)
    assert float(wanted@np.ones(2,np.float32))<0, wanted
