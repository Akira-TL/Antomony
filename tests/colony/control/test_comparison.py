"""对照的公平入口与世界隔离，不断言神经网络一定获胜。"""
import numpy as np
from mathhackson.colony.control.session import ComparisonSession
from mathhackson.colony.control.rules import RuleController
from mathhackson.colony.protocol import Command
from mathhackson.colony.world import World


def test_rule_world_never_constructs_a_neural_brain(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('普通组不能创建神经网络')
    monkeypatch.setattr('mathhackson.colony.world.Brain', forbidden)
    world=World(7,4,mode='rules')
    assert all(isinstance(a.brain,RuleController) for a in world.ants)


def test_comparison_starts_equal_and_advances_on_one_clock():
    session=ComparisonSession(52,4,warmup=5)
    session.dispatch(Command(kind='compare',value=1))
    reference=session.reference
    assert reference is not None
    np.testing.assert_array_equal([a.position for a in session.world.ants],[a.position for a in reference.ants])
    assert not np.shares_memory(session.world.field.values,reference.field.values)
    session.dispatch(Command(kind='wind',value=1.2))
    for _ in range(20): session.tick()
    assert session.world.tick_count==reference.tick_count==20
    assert session.world.wind==reference.wind==1.2
    assert all(a.brain.updates==0 for a in reference.ants)
    assert all(a.brain.updates==20 for a in session.world.ants)
    assert all(isinstance(a.brain,RuleController) for a in reference.ants)
    session.dispatch(Command(kind='pause'))
    session.tick()
    assert session.world.tick_count==reference.tick_count==20
    session.dispatch(Command(kind='step'))
    assert session.world.tick_count==reference.tick_count==21


def test_rejected_edit_is_atomic_in_both_worlds():
    session=ComparisonSession(52,4,warmup=5)
    session.dispatch(Command(kind='compare',value=1))
    session.dispatch(Command(kind='wall',x=-10,y=0))
    assert all(len(w.walls)==2 for w in session.worlds)
    session.dispatch(Command(kind='wall',x=0,y=0,hx=.3,hy=.8,angle=.35))
    assert all(len(w.walls)==3 for w in session.worlds)
    assert session.world.walls[-1] == session.reference.walls[-1]
    session.world.field.values[1,5,5]=1
    assert session.reference.field.values[1,5,5] == 0
