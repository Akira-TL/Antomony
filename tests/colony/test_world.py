"""独立世界、真实动作、操控与学习时序的验收。"""
import numpy as np
from mathhackson.colony.world import World
from mathhackson.colony.protocol import Command,apply_command,snapshot


def test_world_has_independent_brains_and_actual_updates():
    w=World(13,4,warmup=80)
    before=[a.brain.fingerprint() for a in w.ants]
    w.ants[0].brain.frozen=True
    for _ in range(12): w.tick()
    assert w.ants[0].brain.fingerprint()==before[0]
    assert all(a.brain.updates==12 for a in w.ants[1:])
    assert any(a.brain.fingerprint()!=before[a.id] for a in w.ants[1:])
    assert w.samples==48
    frame=snapshot(w)
    assert len(frame.ants)==4 and len(frame.pheromones)>1000


def test_same_seed_and_commands_are_deterministic():
    a,b=World(23,4,warmup=40),World(23,4,warmup=40)
    a.wind=b.wind=.6
    for _ in range(10): a.tick(); b.tick()
    for x,y in zip(a.ants,b.ants):
        np.testing.assert_array_equal(x.position,y.position)
        np.testing.assert_array_equal(x.brain.head.effective,y.brain.head.effective)
    assert not np.shares_memory(a.field.values,b.field.values)


def test_wall_cannot_spawn_inside_ant_or_nest():
    w=World(3,4,warmup=5); count=len(w.walls)
    text=w.add_wall(-10,0)
    assert '重叠' in text and len(w.walls)==count
    assert w.add_wall(0,-6,.3,1)=='墙已放置'
    assert w.remove_wall(0,-6)=='墙已移除'


def test_pause_single_step_and_field_clear_do_not_fake_learning():
    w=World(5,4,warmup=5); w=apply_command(w,Command(kind='pause'))
    assert w.paused
    w=apply_command(w,Command(kind='step')); assert w.tick_count==1
    p=w.ants[0].brain.head.effective.copy()
    w=apply_command(w,Command(kind='clear'))
    np.testing.assert_array_equal(w.ants[0].brain.head.effective,p)
    assert w.field.values.sum()==0


def test_sensor_rays_keep_the_pose_that_actually_produced_them():
    w=World(9,4,warmup=5)
    position=w.ants[0].position.copy(); heading=w.ants[0].heading
    w.tick(); frame=snapshot(w)
    assert frame.ants[0].sense_x==float(position[0])
    assert frame.ants[0].sense_y==float(position[1])
    assert frame.ants[0].sense_heading==heading


def test_neural_motion_predictions_affect_actual_action_choice():
    a,b=World(37,4,warmup=80),World(37,4,warmup=80)
    for ant in a.ants+b.ants: ant.brain.frozen=True
    # 相同感知、规则和随机流，仅反转一个体对候选运动的预测。
    b.ants[0].brain.head.stable[:,:2]*=-1
    a.tick(); b.tick()
    assert a.ants[0].action!=b.ants[0].action


def test_invalid_commands_are_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError): Command(kind='wall',x=float('nan'))
    with pytest.raises(ValidationError): Command(kind='reset',count=900)
