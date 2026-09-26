import hashlib

import pytest

from mathhackson.training.comparison.auditing import verify_manifest
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.foraging.plastic_course.checkpoint import save_checkpoint
from mathhackson.training.foraging.plastic_course.run import Initialization, Protocol, Source
from scripts.analyses.feedback_plasticity import audit_snapshots, freeze_inputs


def test_manifest_is_fixed_outside_inputs_and_detects_later_change(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    file = directory / "record.txt"
    file.write_text("original")
    manifest = tmp_path / "inputs.sha256"
    freeze_inputs(directory, manifest)
    assert verify_manifest(directory, manifest) == 1
    with pytest.raises(FileExistsError):
        freeze_inputs(directory, manifest)
    with pytest.raises(ValueError, match="之外"):
        freeze_inputs(directory, directory / "invalid.sha256")
    file.write_text("changed")
    with pytest.raises(ValueError, match="散列"):
        verify_manifest(directory, manifest)


def test_snapshot_audit_requires_all_versions_and_identical_frozen_motor(tmp_path):
    motor = DirectionMotor(7).freeze()
    motor_path = tmp_path / "motor.npz"
    save_motor(motor_path, motor, MotorSnapshot(seed=7, updates=0))
    source = Source(path=str(motor_path), sha256=hashlib.sha256(motor_path.read_bytes()).hexdigest())
    unused = Source(path="unused-synthetic-foundation.npz", sha256="0" * 64)
    plan = Protocol(motor=source, updates=25, initializations=(
        Initialization(seed=11, foundation=unused, train_seed=100, evaluation_seed=200),
        Initialization(seed=12, foundation=unused, train_seed=300, evaluation_seed=400)))
    directory = tmp_path / "raw"
    for item in plan.initializations:
        model = PlasticDirection(motor, seed=item.seed, max_step=plan.max_step, max_fast=plan.max_fast)
        for update in (0, 25):
            save_checkpoint(directory / f"seed-{item.seed}/step-{update:04d}.npz", model, update=update, seed=item.seed)
    result = audit_snapshots(directory, plan)
    assert [item.snapshots for item in result] == [2, 2]
    assert all(item.parameter_change_norm == 0 for item in result)
    (directory / "seed-11/step-0025.npz").unlink()
    with pytest.raises(ValueError, match="保存点缺失"):
        audit_snapshots(directory, plan)
    broken = PlasticDirection(DirectionMotor(9), seed=11, max_step=plan.max_step, max_fast=plan.max_fast)
    save_checkpoint(directory / "seed-11/step-0025.npz", broken, update=25, seed=11)
    with pytest.raises(ValueError, match="动作底座被改变"):
        audit_snapshots(directory, plan)
