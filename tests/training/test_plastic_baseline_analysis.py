"""独立试行基线的固定判据与短合成整链核验。"""
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
from mathhackson.training.foraging.plastic_course.assessment import SeedAssessment
from mathhackson.training.foraging.plastic_course.auditing import freeze_inputs
from mathhackson.training.foraging.plastic_course.checkpoint import save_checkpoint
from mathhackson.training.foraging.plastic_course.rollout import rollout
from mathhackson.training.foraging.plastic_course.run import (
    Completion, Initialization, Protocol, Source, TrainingRow, base_directions,
    evaluate, save_trace, training_episode,
)
from scripts.analyses.plastic_baseline import Difference, audit_pairing, audit_training_pairs, decide, run


def test_all_three_conditions_are_required_without_erasing_negative_values():
    original = SeedAssessment(initialization=1, contrasts=[], accepted_fraction=.5,
        nonzero_writes=1, association_gain=True, reference_preserved=True,
        learned_timing_gain=True, mixed_control=True, passed=True)
    contrasts = [Difference(initialization=1, condition=condition, comparison=comparison,
        after=[value], last=[value], mean_after=value, mean_last=value)
        for comparison in ("baseline", "training")
        for condition, value in (("association", .03), ("reference", -.02), ("transient", -.1))]
    assert decide(original, contrasts).passed
    contrasts[0] = contrasts[0].model_copy(update={"mean_last": .029})
    assert not decide(original, contrasts).passed
    assert contrasts[2].mean_after == -.1
    assert not decide(original.model_copy(update={"passed": False}), contrasts).passed
    with pytest.raises(ValueError, match="完整且唯一"):
        decide(original, contrasts[:-1])


@pytest.fixture
def records(tmp_path: Path):
    def source(path: str) -> Source:
        return Source(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    motor_source = source("models/interactive/motor/update-001200.npz")
    foundation = source("models/interactive/memory/episode-0000-ant-00.npz")
    plan = Protocol(motor=motor_source, updates=1, evaluation_batches=1, evaluate_initial=True,
        paired_courses=True, direction_mode="bounded", course=CueConfig(steps=8, batch=2), initializations=(
            Initialization(seed=41, foundation=foundation, train_seed=100, evaluation_seed=200),
            Initialization(seed=42, foundation=foundation, train_seed=300, evaluation_seed=400)))
    root = tmp_path / "raw"
    root.mkdir()
    motor, _ = load_motor(Path(motor_source.path))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for mode in ("history", "paired"):
            current = plan.model_copy(update={"baseline_mode": mode})
            folder = root / mode
            folder.mkdir()
            (folder / "protocol.json").write_text(current.model_dump_json(), encoding="utf-8")
            for init in current.initializations:
                base = MemoryPolicy.load(Path(foundation.path))
                model = PlasticDirection(motor, seed=init.seed, direction_mode="bounded",
                    max_step=plan.max_step, max_fast=plan.max_fast)
                for stage, update in ((Path(), 1), (Path("untrained"), 0)):
                    output = folder / stage / f"seed-{init.seed}"
                    output.mkdir(parents=True)
                    save_checkpoint(output / "step-0000.npz", model, update=0, seed=init.seed)
                    if update:
                        save_checkpoint(output / "step-0001.npz", model, update=1, seed=init.seed)
                        episode = training_episode(51, current)
                        direction = base_directions(base, episode)
                        with torch.no_grad():
                            trace = rollout(model, episode, direction, seed=52, training=True)
                        save_trace(output / "training-0001.npz", episode, direction, trace, structure_update=0)
                        row = TrainingRow(update=1, loss=0, mean_reward=0, accepted_fraction=.5,
                            gradient_norm=0, elapsed_seconds=0)
                        (output / "training.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")
                    evaluate(copy.deepcopy(model), base, init, current, output,
                        time.monotonic() + 30, structure_update=update)
            (folder / "completion.json").write_text(Completion(complete=True, reason="合成夹具，不优化参数",
                elapsed_seconds=0, initializations_completed=2).model_dump_json(), encoding="utf-8")
    finally:
        torch.set_num_threads(threads)
    manifest = tmp_path / "inputs.sha256"
    freeze_inputs(root, manifest)
    return root, plan, manifest


def test_complete_short_records_are_audited(records):
    root, _, manifest = records
    result = run(root, root / "history/protocol.json", root / "paired/protocol.json", manifest)
    assert result.evaluation_frames == 1536
    assert len(result.contrasts) == 12 and len(result.decisions) == 2
    assert not result.all_passed


def test_off_control_cannot_change_between_baselines(records):
    root, plan, _ = records
    path = root / "paired/seed-41/reference-200-off.npz"
    with np.load(path, allow_pickle=False) as original:
        values = {key: original[key] for key in original.files}
    values["probabilities"][0, 0, 0] += .01
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="跨基线/时点改变"):
        audit_pairing(root, plan)


def test_training_pair_must_share_external_course(records):
    root, plan, _ = records
    path = root / "paired/seed-41/training-0001.npz"
    with np.load(path, allow_pickle=False) as original:
        values = {key: original[key] for key in original.files}
    values["targets"][0, 1, 0] += .1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="同一情境"):
        audit_training_pairs(root / "paired", plan)
