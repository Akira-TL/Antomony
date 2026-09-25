"""神经参数隔离与局部训练的机械检查。"""
import numpy as np
from mathhackson.colony.brain import Brain


def test_independent_initialization_and_online_update():
    a,b=Brain(31),Brain(32)
    before=b.head.effective.copy()
    assert a.fingerprint()!=b.fingerprint()
    assert not np.shares_memory(a.w1,b.w1)
    x,y=a.practice(16); a.train(x,y)
    np.testing.assert_array_equal(b.head.effective,before)
    assert a.updates==1 and b.updates==0


def test_freezing_keeps_parameters_exactly():
    a=Brain(21); x,y=a.practice(32); before=a.head.effective.copy()
    a.frozen=True; a.train(x,y)
    np.testing.assert_array_equal(a.head.effective,before)
    assert a.updates==0


def test_independent_warmup_reduces_held_out_local_error():
    for seed in [7,13,29]:
        a=Brain(seed)
        assert a.warm_loss < a.birth_loss*.6
        assert np.isfinite(a.head.effective).all()
