import numpy as np
import pytest
import torch
from dataclasses import replace
from pathlib import Path

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


def test_training_teacher_keeps_real_state_and_equal_branches_only_pay_write_cost(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/training"))
    from direction_gate import Episode, Individual, teacher_benefit, WRITE_COST
    from mathhackson.training.direction.environment import DirectionEnvironment
    model = DirectionCorrection(DirectionMotor(41))
    optimizer = torch.optim.SGD(model.plastic_parameters(), lr=.05)
    env = DirectionEnvironment(.2, .8, 1., 2.)
    original_env = replace(env)
    individual = Individual(model, optimizer, env, FeedbackHistory(),
                            [p.detach().clone() for p in model.motor.parameters()])
    episode = Episode(motor_seed=41, scene_seed=4101, scenario="positive", magnitude=.15)
    benefit = teacher_benefit(individual, episode, 70, proposed=0.)
    assert benefit == pytest.approx(-WRITE_COST, abs=1e-12)
    assert env == original_env
    assert model.offset.item() == 0.
    individual.assert_frozen()
    individual.observe_and_propose(.8, .15, 71)
    assert model.offset.item() == 0.
    individual.assert_frozen()


def test_random_control_preserves_exact_acceptance_budget_and_reproducibility(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/training"))
    from direction_gate_control import random_schedule
    for count in (0, 7, 192):
        first = random_schedule(192, count, (6101, 41, 5201, 101, 0))
        second = random_schedule(192, count, (6101, 41, 5201, 101, 0))
        assert first.dtype == np.bool_
        assert first.shape == (192,)
        assert np.count_nonzero(first) == count
        np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        random_schedule(192, 193, (6101,))


def test_random_control_rejects_invalid_schedule_before_loading_models(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/training"))
    from direction_gate import Episode, evaluate
    episode = Episode(motor_seed=41, scene_seed=5201, scenario="normal", magnitude=.22)
    with pytest.raises(ValueError):
        evaluate(episode, "random", 101, None, tmp_path, schedule=np.zeros(191, dtype=np.bool_))
    with pytest.raises(ValueError):
        evaluate(episode, "random", 101, None, tmp_path, schedule=np.zeros(192))
    with pytest.raises(ValueError):
        evaluate(episode, "learned", 101, None, tmp_path, schedule=np.zeros(192, dtype=np.bool_))
