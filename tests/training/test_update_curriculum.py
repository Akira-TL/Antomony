from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.candidate_probe import ProbePlan, initialize_residual
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.trust_candidate import TrustDirectionLearner


def agent():
    return TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14)


def test_initial_perturbation_is_reproducible_individual_and_not_a_learning_update(tmp_path):
    first, same, other = agent(), agent(), agent()
    random_before = first.random.get_state().clone()
    base = [p.detach().clone() for p in first.policy.parameters()]
    initialize_residual(first, seed=1, norm=.75)
    initialize_residual(same, seed=1, norm=.75)
    initialize_residual(other, seed=2, norm=.75)
    np.testing.assert_array_equal(first.weights(), same.weights())
    assert not np.array_equal(first.weights(), other.weights())
    assert np.linalg.norm(first.parameter.fast) == pytest.approx(.75)
    assert first.writes == first.decisions == 0 and not first.parameter.recent
    assert first.random.get_state().equal(random_before)
    assert all(torch.equal(a, b) for a, b in zip(base, first.policy.parameters(), strict=True))
    path = tmp_path / "initial.npz"
    first.save(path)
    restored = TrustDirectionLearner.load(path, first.motor, 14, first.config)
    np.testing.assert_array_equal(first.weights(), restored.weights())
    with pytest.raises(ValueError):
        initialize_residual(first, seed=3, norm=.5)


def test_zero_default_and_invalid_initialization():
    first = agent()
    initialize_residual(first, seed=1, norm=0.)
    assert not np.any(first.weights())
    for norm in (-1., float("nan"), float("inf"), first.config.maximum_residual_norm):
        with pytest.raises(ValueError):
            initialize_residual(first, seed=1, norm=norm)
    first.decisions = 1
    with pytest.raises(ValueError):
        initialize_residual(first, seed=1, norm=.75)


def test_protocol_preserves_default_and_rejects_unsafe_initialization():
    old = ProbePlan.model_validate_json(Path(".research/protocols/trust-candidate-value.json").read_text())
    assert old.initial_residual_norm == 0.
    changed = ProbePlan.model_validate(old.model_dump() | {"initial_residual_norm": .75})
    assert changed.initial_residual_norm == .75
    for norm in (float("nan"), 4.):
        with pytest.raises(ValueError):
            ProbePlan.model_validate(old.model_dump() | {"initial_residual_norm": norm})
