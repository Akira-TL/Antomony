import pytest

from mathhackson.interactive.protocol import SessionConfig
from mathhackson.interactive.session import LiveSession, make_plan
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.update_decision import UpdateDecision


@pytest.fixture
def live(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for i in range(2):
        MemoryPolicy(FeedforwardPolicy(81 + i)).save(sources / f"episode-0000-ant-{i:02d}.npz", update=0, phase="frozen")
        FeedforwardPolicy(81 + i, hidden_width=17).save(sources / f"seed-{81+i}-signal-002400.npz", update=2400, phase="signal")
        (sources / f"ant-{i:02d}").mkdir()
        decision = UpdateDecision()
        decision.updates = 200
        decision.save(sources / f"ant-{i:02d}/step-0200.npz")
    save_motor(sources / "motor.npz", DirectionMotor(41), MotorSnapshot(seed=41, updates=0))
    config = SessionConfig(ants=2, horizon=16, stock=48)
    plan = make_plan(config).model_copy(update={"policy_directory": str(sources), "gate_directory": str(sources),
        "mlp_directory": str(sources), "motor": str(sources / "motor.npz"), "checkpoint_every": 8})
    session = LiveSession(config, tmp_path / "run", plan=plan)
    yield session
    session.close()
