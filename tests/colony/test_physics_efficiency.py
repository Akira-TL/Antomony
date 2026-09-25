"""空旷场景不反复进行已经收敛的圆形接触求解。"""
import numpy as np
from mathhackson.colony import geometry


def test_no_contact_solver_stops_after_convergence(monkeypatch):
    calls=0
    original=geometry.project_static
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(geometry,'project_static',counted)
    positions=np.asarray([[0.,0.],[4.,4.]],np.float32)
    motion=np.asarray([[.05,0.],[0.,.05]],np.float32)
    moved,contact=geometry.move_discs(positions,motion,.2,[],np.asarray([14.,10.],np.float32))
    np.testing.assert_allclose(moved,positions+motion)
    assert not contact.any()
    assert calls<=4, calls
