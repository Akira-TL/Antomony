from pathlib import Path

import pytest
import torch

from mathhackson.training.motor_checkpoint import (MotorCheckpoint,
                                                   load_motor_checkpoint)
from mathhackson.training.recurrent import RecurrentPolicy
from mathhackson.training.recurrent_session import RecurrentSession

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("episode", [568, 712])
def test_saved_motor_checkpoint_restores_only_four_connections(episode: int):
    path = ROOT / "checkpoints" / "recurrent" / f"motor-episode-{episode:06d}.json"
    saved = MotorCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    policy = RecurrentPolicy(123)
    loaded = load_motor_checkpoint(policy, path)
    assert loaded == saved
    assert loaded.source_episode == episode
    assert torch.equal(policy.motor, saved.motor_matrix())
    assert torch.count_nonzero(policy.motor) == 4
    assert policy.phase == "memory"
    assert torch.count_nonzero(policy.trainable_masks()[0]) == 0
    assert policy.hidden_history == []


def test_motor_checkpoint_cannot_replace_trained_policy():
    policy = RecurrentPolicy(123)
    policy.outer_updates = 1
    path = ROOT / "checkpoints" / "recurrent" / "motor-episode-000568.json"
    with pytest.raises(ValueError, match="尚未训练"):
        load_motor_checkpoint(policy, path)


def test_new_session_can_continue_motor_training_from_saved_weights(tmp_path: Path):
    path = ROOT / "checkpoints" / "recurrent" / "motor-episode-000568.json"
    session = RecurrentSession(tmp_path, motor_checkpoint=path, continue_motor=True)
    saved = MotorCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    state = session.state()
    assert state.paused
    assert state.phase == "motor"
    assert state.episode == 1
    assert state.motor_source_episode == 568
    assert state.motor_source_session == saved.source_session
    assert torch.equal(session.model.motor, saved.motor_matrix())
    assert torch.count_nonzero(session.model.trainable_masks()[0]) == 4
    assert session.motor_alignment() == (6, 6)
