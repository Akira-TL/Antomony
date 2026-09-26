import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from mathhackson.training.comparison.acceptance_audit import audit_inputs
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.update_curriculum import CurriculumPlan, collect_curriculum


def manifest(directory, path):
    path.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n"
                            for p in sorted(directory.rglob("*")) if p.is_file()))


@pytest.mark.parametrize("tamper", [None, "weights", "frames", "feedback"])
def test_short_real_dataset_audits_frozen_weights_frames_and_feedback(tmp_path, tamper):
    base = CurriculumPlan.model_validate_json(Path(".research/protocols/memory-update-learning.json").read_text())
    fields = base.model_dump()
    fields.update(train_seeds=(1,), held_seeds=(2,))
    fields["probe"].update(seeds=(1, 2), policy_directory=str(tmp_path), motor_path=str(tmp_path / "motor.npz"))
    fields["probe"]["environment"].update(ants=1, horizon=8)
    fields["probe"]["adaptation"]["window"] = 4
    plan = CurriculumPlan.model_validate(fields)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(plan.model_dump_json())
    MemoryPolicy(FeedforwardPolicy(1)).save(tmp_path / "episode-0000-ant-00.npz", update=0, phase="frozen")
    save_motor(tmp_path / "motor.npz", DirectionMotor(41), MotorSnapshot(seed=41, updates=0))
    directory, listing = tmp_path / "data", tmp_path / "manifest.sha256"
    collect_curriculum(plan, directory, plan_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    target = directory / "initial-0-seed-1"
    if tamper == "frames":
        with gzip.open(target / "parent.jsonl.gz", "rt") as stream:
            lines = stream.readlines()
        with gzip.open(target / "parent.jsonl.gz", "wt") as stream:
            stream.writelines(lines[:-1])
    elif tamper in ("weights", "feedback"):
        path = target / ("tick-0008-ant-00.npz" if tamper == "weights" else "tick-0008-ant-00.residual.npz")
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        if tamper == "weights":
            values["recent_weights"][0, 0] += .1
        else:
            values["config_json"] = plan.probe.adaptation.model_copy(update={"feedback_mode": "critic"}).model_dump_json()
        np.savez(path, **values)
    manifest(directory, listing)
    if tamper:
        with pytest.raises(ValueError):
            audit_inputs(directory, listing, protocol)
    else:
        execution, audit = audit_inputs(directory, listing, protocol)
        assert execution.plan == plan and audit.snapshots == 6 and audit.frames == 32
