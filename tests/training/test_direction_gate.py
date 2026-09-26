import numpy as np
import pytest
import torch

from mathhackson.training.direction.gate import FeedbackHistory, Feedback, UpdateGate, accept_proposal
from mathhackson.training.direction.adaptation import DirectionCorrection
from mathhackson.training.direction.policy import DirectionMotor


def test_history_keeps_both_recent_and_sparse_taps_with_missing_masks():
    history = FeedbackHistory()
    empty = history.features(0., 0.)
    assert empty.shape == (34,)
    assert not empty.any()
    for index in range(1, 18):
        history.append(Feedback(error=index / 20., turn=0., residual=0.))
    features = history.features(.07, .08).reshape(-1)
    blocks = features[2:].reshape(8, 4)
    np.testing.assert_allclose(blocks[:, 0], np.array([17, 16, 15, 14, 14, 10, 6, 2]) / 20. / np.pi)
    np.testing.assert_array_equal(blocks[:, 3], np.ones(8))
    assert features[0] == pytest.approx(.2)
    assert features[1] == pytest.approx(.2)


def test_rejected_proposal_does_not_write_model_parameters():
    model = DirectionCorrection(DirectionMotor(41))
    before = [p.detach().clone() for p in model.parameters()]
    accept_proposal(model, .1, False)
    assert all(a.equal(b) for a, b in zip(before, model.parameters(), strict=True))
    accept_proposal(model, .1, True)
    assert model.offset.item() == pytest.approx(.1)
    assert all(a.equal(b) for a, b in zip(before[1:], list(model.parameters())[1:], strict=True))


def test_gate_has_no_environment_argument_and_saves_exact_weights(tmp_path):
    model = UpdateGate(91)
    features = FeedbackHistory().features(0., .01)
    expected = model.probability(features)
    path = tmp_path / "gate.npz"
    model.save(path, update=100)
    restored = UpdateGate.load(path)
    assert restored.probability(features) == expected
    with pytest.raises(FileExistsError):
        model.save(path, update=200)
    assert all(not p.requires_grad for p in restored.parameters())


def test_gate_rejects_nonfinite_input_and_out_of_bounds_proposal():
    gate = UpdateGate(8)
    with pytest.raises(ValueError):
        gate.probability(np.full(34, np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        accept_proposal(DirectionCorrection(DirectionMotor(8)), 1., True)
