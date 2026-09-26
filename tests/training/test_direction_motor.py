import math
from pathlib import Path

import numpy as np
import pytest

from mathhackson.training.direction.environment import DirectionEnvironment
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.direction.checkpoint import MotorSnapshot, load_motor, save_motor


def test_body_relative_direction_is_independent_of_position():
    env = DirectionEnvironment(heading=math.pi / 2, desired=math.pi)
    np.testing.assert_allclose(env.observation(), [0., 1.], atol=1e-7)
    env.x, env.y = 123., -456.
    np.testing.assert_allclose(env.observation(), [0., 1.], atol=1e-7)


def test_turning_without_moving_changes_heading_but_not_position():
    env = DirectionEnvironment(heading=0., desired=math.pi / 2)
    env.step(False, 1.)
    assert env.heading == pytest.approx(math.pi / 18)
    assert (env.x, env.y) == (0., 0.)
    env.step(False, -1.)
    assert env.heading == pytest.approx(0.)


def test_frozen_direction_policy_rejects_old_observations_and_keeps_weights():
    policy = DirectionMotor(seed=17).freeze()
    with pytest.raises(ValueError, match="两个"):
        policy.decide(np.zeros(16, dtype=np.float32))
    with pytest.raises(ValueError, match="单位"):
        policy.decide(np.array([0., 0.], dtype=np.float32))
    with pytest.raises(ValueError, match="有限"):
        policy.decide(np.array([np.nan, 1.], dtype=np.float32))
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    first = policy.decide(np.array([0., 1.], dtype=np.float32))
    second = policy.decide(np.array([0., 1.], dtype=np.float32))
    assert first == second
    assert isinstance(first.move, bool)
    assert -1. <= first.turn <= 1.
    for original, parameter in zip(before, policy.parameters(), strict=True):
        assert not parameter.requires_grad
        assert parameter.equal(original)


def test_checkpoint_roundtrip_is_exact_and_does_not_alias(tmp_path: Path):
    source = DirectionMotor(seed=22).freeze()
    snapshot = MotorSnapshot(seed=22, updates=0)
    path = tmp_path / "initial.npz"
    save_motor(path, source, snapshot)
    loaded, metadata = load_motor(path)
    assert metadata == snapshot
    for original, restored in zip(source.parameters(), loaded.parameters(), strict=True):
        assert restored.equal(original)
        assert restored.data_ptr() != original.data_ptr()
        assert not restored.requires_grad
    assert source.decide(np.array([1., 0.], dtype=np.float32)) == loaded.decide(
        np.array([1., 0.], dtype=np.float32))
    with pytest.raises(FileExistsError):
        save_motor(path, source, snapshot)
