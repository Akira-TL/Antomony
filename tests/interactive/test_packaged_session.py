from pathlib import Path

from mathhackson.interactive.protocol import SessionConfig
from mathhackson.interactive.session import LiveSession


def test_default_session_only_loads_tracked_model_package(tmp_path):
    session = LiveSession(SessionConfig(ants=2, horizon=16), tmp_path / "packaged")
    try:
        paths = [session.plan.motor, str(session.plan.policy_path(0)), str(session.plan.gate_path(0)), str(session.plan.mlp_path(0))]
        assert all(Path(p).is_relative_to("models/interactive") for p in paths)
        for _ in range(20):
            session.advance()
        assert not session.done and session.paused
        assert [g.world.steps for g in session.groups] == [20, 20, 20]
    finally:
        session.close()
