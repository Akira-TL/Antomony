"""放墙事务的空间、参数与路径积分约束。"""
import numpy as np
from mathhackson.colony.world import World


def test_wall_can_displace_an_ant_without_training_or_overlap():
    w=World(37,1,warmup=5)
    w.ants[0].position=np.zeros(2,np.float32)
    w.ants[0].home=w.nest-w.ants[0].position
    a=w.ants[0]
    before=a.brain.fingerprint()
    result=w.add_wall(0.,0.,.4,2.)
    assert '已放置' in result, result
    assert not w.walls[-1].overlaps(a.position,w.radius)
    assert a.brain.fingerprint()==before and a.brain.updates==0
    np.testing.assert_allclose(a.home,w.nest-a.position,atol=1e-6)


def test_preview_is_read_only_and_crowded_placement_has_no_overlap():
    w=World(22,8,warmup=5)
    for i,a in enumerate(w.ants):
        a.position=np.asarray([0.,(i-3.5)*.44],np.float32)
        a.home=w.nest-a.position
    before=np.stack([a.position for a in w.ants])
    fingerprints=[a.brain.fingerprint() for a in w.ants]
    wall,positions,_=w.plan_wall(0,0,.4,2.)
    assert wall is not None and positions is not None
    np.testing.assert_array_equal(before,np.stack([a.position for a in w.ants]))
    assert len(w.walls)==2
    assert '已放置' in w.add_wall(0,0,.4,2.)
    for i,a in enumerate(w.ants):
        assert not any(v.overlaps(a.position,w.radius) for v in w.walls)
        assert a.brain.fingerprint()==fingerprints[i]
        np.testing.assert_allclose(a.home,w.nest-a.position,atol=1e-6)
        for b in w.ants[:i]: assert np.linalg.norm(a.position-b.position)>=2*w.radius-1e-5


def test_invalid_wall_keeps_world_unchanged():
    w=World(23,4,warmup=5)
    positions=np.stack([a.position for a in w.ants])
    field=w.field.values.copy()
    for x,y in [(-10.,0.),(13.9,0.),(7.,-4.)]:
        assert '拒绝' in w.add_wall(x,y)
        assert len(w.walls)==2
        np.testing.assert_array_equal(positions,np.stack([a.position for a in w.ants]))
        np.testing.assert_array_equal(field,w.field.values)

