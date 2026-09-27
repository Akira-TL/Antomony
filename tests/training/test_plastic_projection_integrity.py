"""短合成记录的整链审计；不训练，也不接触正式评价分区。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.feedback_cues import CueConfig
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.foraging.plastic_course.auditing import freeze_inputs
from mathhackson.training.foraging.plastic_course.checkpoint import save_checkpoint
from mathhackson.training.foraging.plastic_course.run import Completion, Initialization, Protocol, Source, evaluate
from scripts.analyses.plastic_projection import audit_pairing, run


Records = tuple[Path, Protocol, Path, Path, Path]


@pytest.fixture
def paired_records(tmp_path: Path) -> Records:
    def source(path: str) -> Source:
        return Source(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    motor_source = source("models/interactive/motor/update-001200.npz")
    foundation = source("models/interactive/memory/episode-0000-ant-00.npz")
    plan = Protocol(motor=motor_source, updates=1, evaluation_batches=1, evaluate_initial=True,
        course=CueConfig(steps=8, batch=1), initializations=(
            Initialization(seed=21, foundation=foundation, train_seed=100, evaluation_seed=200),
            Initialization(seed=22, foundation=foundation, train_seed=300, evaluation_seed=400)))
    root = tmp_path / "raw"
    root.mkdir()
    motor, _ = load_motor(Path(motor_source.path))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for mode in ("unit", "bounded"):
            current = plan.model_copy(update={"direction_mode": mode})
            folder = root / mode
            folder.mkdir()
            (folder / "protocol.json").write_text(current.model_dump_json(), encoding="utf-8")
            for initialization in plan.initializations:
                base = MemoryPolicy.load(Path(foundation.path))
                model = PlasticDirection(motor, seed=initialization.seed, direction_mode=mode,
                    max_step=plan.max_step, max_fast=plan.max_fast)
                for stage, update in ((Path(), 1), (Path("untrained"), 0)):
                    output = folder / stage / f"seed-{initialization.seed}"
                    output.mkdir(parents=True)
                    save_checkpoint(output / "step-0000.npz", model, update=0, seed=initialization.seed)
                    if update:
                        save_checkpoint(output / "step-0001.npz", model, update=1, seed=initialization.seed)
                    evaluate(copy.deepcopy(model), base, initialization, current, output,
                        time.monotonic() + 60, structure_update=update)
            (folder / "completion.json").write_text(Completion(complete=True, reason="合成审计夹具，不训练",
                elapsed_seconds=0, initializations_completed=2).model_dump_json(), encoding="utf-8")
    finally:
        torch.set_num_threads(previous_threads)
    manifest = tmp_path / "inputs.sha256"
    freeze_inputs(root, manifest)
    return root, plan, root / "unit/protocol.json", root / "bounded/protocol.json", manifest


def test_complete_comparison_records_are_audited(paired_records: Records) -> None:
    root, _, unit, bounded, manifest = paired_records
    result = run(root, unit, bounded, manifest)
    assert result.evaluation_frames == 768
    assert len(result.evaluations) == 4
    assert len(result.contrasts) == 12
    assert not result.all_passed


def test_initial_reference_must_have_initial_structure_identity(paired_records: Records) -> None:
    root, plan, *_ = paired_records
    path = root / "bounded/untrained/seed-21/step-0000.npz"
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["update"] = np.asarray(1)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="训练前参照的结构身份不匹配"):
        audit_pairing(root, plan)


def test_off_control_must_match_across_architectures(paired_records: Records) -> None:
    root, plan, *_ = paired_records
    path = root / "bounded/seed-21/reference-200-off.npz"
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["probabilities"][0, 0, 0] += .01
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="跨架构/时点不一致"):
        audit_pairing(root, plan)
