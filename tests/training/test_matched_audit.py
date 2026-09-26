import hashlib

import pytest
import torch

from mathhackson.training.comparison.matched_audit import MODELS, audit, audit_training, scores
from mathhackson.training.comparison.matched_foundation import MatchedPlan, run
from mathhackson.training.comparison.run import WorldRecord
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.mlp import FeedforwardPolicy


def worlds():
    return [WorldRecord(model=name, seed=seed, sampled=None if name == "rules" else True,
        task=task, steps=100 if name == "mlp-matched" else 120, deliveries=16 if task == "food" else 0,
        pickups=16 if task == "food" else 0, budget_returns=4, returning_individuals=4,
        exhausted=1, mean_max_radius=5., updates=[0] * 8)
        for seed in (1, 2) for task in ("food", "empty") for name in MODELS]


def test_scores_pair_by_seed_keep_negative_and_exclude_incomplete_time():
    rows = worlds()
    plan = MatchedPlan(evaluation_seeds=(1, 2))
    rows[0].deliveries = 8
    rows[3].returning_individuals = 2
    groups, contrasts = scores(list(reversed(rows)), plan)
    assert groups[0].delivery_fraction.mean == .75
    assert groups[0].empty_returning_fraction.mean == .375
    assert not groups[0].minimum_ready
    assert contrasts[0].delivery_fraction.mean == -.25
    assert contrasts[0].complete_pairs == 1
    assert contrasts[0].paired_completion_steps.mean == -20.


@pytest.mark.parametrize("change", ["missing", "duplicate", "invalid"])
def test_scores_reject_invalid_units(change):
    rows = worlds()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[0].returning_individuals = 9
    with pytest.raises(ValueError):
        scores(rows, MatchedPlan(evaluation_seeds=(1, 2)))


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for seed in (81, 82):
        FeedforwardPolicy(seed).save(source / f"seed-{seed}-signal-002400.npz", update=2400, phase="signal")
    motor = source / "motor.npz"
    save_motor(motor, DirectionMotor(41), MotorSnapshot(seed=41, updates=0))
    plan = MatchedPlan(model_seeds=(81, 82), evaluation_seeds=(17999,), signal_updates=2, checkpoint_every=1,
        environment=ColonyConfig(ants=2, horizon=8), motor=str(motor), original_directory=str(source))
    directory = tmp_path / "raw"
    run(plan, directory)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(plan.model_dump_json())
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n" for p in sorted(directory.iterdir())))
    return directory, manifest, protocol, plan


def test_full_audit_checks_actual_training_and_raw_trajectory(sample):
    directory, manifest, protocol, _ = sample
    result = audit(directory, manifest, protocol)
    assert result.files_verified == 21 and result.frames_verified == 48
    assert result.snapshots_verified == 6 and result.optimizers_verified == 4
    assert not result.equivalence_established and not result.all_minimum_ready


def test_optimizer_step_tampering_is_rejected(sample):
    directory, _, _, plan = sample
    path = directory / "seed-81-signal-000002.optimizer.pt"
    state = torch.load(path, weights_only=True)
    state["state"][0]["step"] = torch.tensor(99.)
    torch.save(state, path)
    with pytest.raises(ValueError, match="步数"):
        audit_training(directory, plan)


def test_frozen_value_tampering_is_rejected(sample):
    directory, _, _, plan = sample
    path = directory / "seed-81-signal-000002.npz"
    model = FeedforwardPolicy.load(path)
    with torch.no_grad():
        model.value.bias.add_(1.)
    path.unlink()
    model.save(path, update=2, phase="signal")
    with pytest.raises(ValueError, match="冻结价值"):
        audit_training(directory, plan)
