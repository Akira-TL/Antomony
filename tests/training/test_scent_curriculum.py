from pathlib import Path

import torch

from mathhackson.training.roundtrip_policy import RoundTripPolicy
from mathhackson.training.scent_curriculum import evaluate_scent_reader, pretrain_scent_reader

FOUNDATION = Path(__file__).resolve().parents[2] / "checkpoints/recurrent/foundation-episode-002570.npz"


def test_scent_curriculum_learns_conflicting_local_signals_without_unfreezing_motor():
    model = RoundTripPolicy(91, FOUNDATION)
    before = evaluate_scent_reader(model)
    motor = model.motor.detach().clone()
    input_weights = model.input_weights.detach().clone()
    hidden_weights = model.hidden_weights.detach().clone()
    release_weights = model.release_weights.detach().clone()
    checkpoints = []
    after = pretrain_scent_reader(
        model, steps=300, checkpoint_every=100,
        checkpoint=lambda step, metrics: checkpoints.append((step, metrics.conflict_accuracy)))
    assert after.turn_error < before.turn_error - .5
    assert after.conflict_accuracy > .85
    assert [step for step, _ in checkpoints] == [100, 200, 300]
    assert torch.equal(model.motor, motor)
    assert torch.equal(model.hidden_weights, hidden_weights)
    assert torch.equal(model.release_weights, release_weights)
    assert torch.equal(model.input_weights[:, :8], input_weights[:, :8])
    assert torch.equal(model.input_weights[:, 14:16], input_weights[:, 14:16])
    assert not torch.equal(model.input_weights[:, 8:14], input_weights[:, 8:14])
