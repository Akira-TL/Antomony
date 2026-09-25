from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.roundtrip_session import RoundTripSession
from mathhackson.training.schemas import RoundTripCommand

FOUNDATION = Path(__file__).resolve().parents[2] / "checkpoints/recurrent/foundation-episode-002570.npz"


def test_roundtrip_session_starts_paused_and_records_both_release_channels(tmp_path):
    session = RoundTripSession(tmp_path, FOUNDATION, seed=4)
    initial_motor = session.model.motor.detach().clone()
    assert session.paused and session.model.phase == "memory"
    assert session.model.write_mode == "off"
    assert session.state().field_width == 96
    session.env.horizon = 2
    session.command(RoundTripCommand(action="step"))
    assert session.state().steps == 1
    session.command(RoundTripCommand(action="step"))
    assert session.episode == 2
    assert len(session.history) == 1
    assert session.history[0].completed
    with np.load(session.directory / session.history[0].checkpoint, allow_pickle=False) as archive:
        assert archive["actions"].shape == (2, 4)
        assert archive["observations"].shape == (2, 17)
        assert archive["release_weights"].shape == (2, 4)
    assert torch.equal(session.model.motor, initial_motor)


def test_phase_change_preserves_parameters_but_resets_episode_state(tmp_path):
    session = RoundTripSession(tmp_path, FOUNDATION, seed=5)
    session.command(RoundTripCommand(action="step"))
    motor = session.model.motor.detach().clone()
    session.command(RoundTripCommand(action="phase", phase="adaptive"))
    assert session.paused and session.env.steps == 0
    assert session.model.phase == "adaptive" and session.model.write_mode == "learned"
    assert torch.equal(session.model.motor, motor)
    assert session.history[0].steps == 1
    assert not session.history[0].completed
    with pytest.raises(ValueError, match="停止外部训练"):
        session.command(RoundTripCommand(action="write_mode", write_mode="always"))
    session.command(RoundTripCommand(action="phase", phase="autonomous"))
    session.command(RoundTripCommand(action="write_mode", write_mode="always"))
    assert session.model.write_mode == "always"
