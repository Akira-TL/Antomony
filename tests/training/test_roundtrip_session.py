from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.roundtrip_environment import RoundTripEnvironment
from mathhackson.training.roundtrip_policy import (ROUNDTRIP_MODEL_VERSION,
                                                  ROUNDTRIP_NAMES, RoundTripPolicy)
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


def test_demo_snapshots_replay_same_seed_without_training(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    original = RoundTripPolicy(6, FOUNDATION)
    source = tmp_path / "scent.npz"
    np.savez_compressed(source, model_version=ROUNDTRIP_MODEL_VERSION,
                        **{name: parameter.detach().numpy() for name, parameter in
                           zip(ROUNDTRIP_NAMES, original.parameters, strict=True)})
    for step, value in ((25, .25), (100, 1.)):
        with torch.no_grad():
            original.action_weights[1, 0] = value
        np.savez_compressed(demo / f"teacher-step-{step:06d}.npz",
                            model_version=ROUNDTRIP_MODEL_VERSION,
                            **{name: parameter.detach().numpy() for name, parameter in
                               zip(ROUNDTRIP_NAMES, original.parameters, strict=True)})

    session = RoundTripSession(tmp_path / "sessions", FOUNDATION, seed=6,
                               demo_dir=demo, source_checkpoint=source)
    assert session.paused
    assert session.state().scene_seed == 0
    assert session.state().snapshot_id == "teacher-step-000100"
    assert [item.id for item in session.state().snapshots] == [
        "foundation", "scent-base", "teacher-step-000025", "teacher-step-000100"]
    assert session.model.phase == "autonomous" and session.model.write_mode == "off"
    food = session.env.food.copy()
    session.command(RoundTripCommand(action="step"))
    session.command(RoundTripCommand(action="snapshot", snapshot_id="teacher-step-000025"))
    assert session.paused and session.env.steps == 0
    assert np.array_equal(session.env.food, food)
    assert session.model.action_weights[1, 0] == .25
    assert session.history[-1].completed is False
    session.command(RoundTripCommand(action="reset", seed=17))
    assert session.state().scene_seed == 17
    assert np.array_equal(session.env.food, RoundTripEnvironment(17).food)
    assert session.state().snapshot_id == "teacher-step-000025"
    with pytest.raises(ValueError, match="未找到"):
        session.command(RoundTripCommand(action="snapshot", snapshot_id="outside.npz"))
    assert session.state().snapshot_id == "teacher-step-000025"
    session.env.horizon = 1
    session.command(RoundTripCommand(action="step"))
    assert session.state().scene_seed == 18
    assert np.array_equal(session.env.food, RoundTripEnvironment(18).food)
