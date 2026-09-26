import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.candidate_probe import initialize_residual
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.trust_candidate import TrustDirectionLearner


def learner():
    return TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14)


def test_coherent_initialization_keeps_norm_and_changes_only_one_receiver():
    first, same = learner(), learner()
    initialize_residual(first, seed=1, norm=.75, pattern="coherent-fourth")
    initialize_residual(same, seed=1, norm=.75, pattern="coherent-fourth")
    np.testing.assert_array_equal(first.weights(), same.weights())
    weights = first.weights().reshape(9, 5)
    assert np.linalg.norm(weights) == pytest.approx(.75)
    assert not weights[:, 1:].any() and np.all(weights[:, 0] == weights[0, 0])
    assert first.decisions == first.writes == 0
    with pytest.raises(ValueError):
        initialize_residual(learner(), seed=1, norm=0., pattern="invalid")


def test_restoration_changes_only_clone_and_restores_stable_not_arbitrary_zero():
    source = learner()
    source.parameter.stable[:] = .01
    initialize_residual(source, seed=1, norm=.75, pattern="coherent-fourth")
    before = source.weights()
    random_before = source.random.get_state().clone()
    source.history.append(torch.ones(8))
    restored = EvaluationActor(source, reset_fast=True)
    assert restored.changed and not restored.agent.parameter.fast.any()
    np.testing.assert_array_equal(restored.agent.weights(), source.parameter.stable)
    np.testing.assert_array_equal(source.weights(), before)
    assert source.random.get_state().equal(random_before)
    restored.agent.history[-1].zero_()
    assert torch.all(source.history[-1] == 1.)
    with pytest.raises(ValueError):
        EvaluationActor(source, accept=True, reset_fast=True)


def test_zero_restoration_is_exact_negative_control():
    source = learner()
    skipped, restored = EvaluationActor(source), EvaluationActor(source, reset_fast=True)
    assert not restored.changed
    np.testing.assert_array_equal(skipped.agent.weights(), restored.agent.weights())
    assert skipped.agent.random.get_state().equal(restored.agent.random.get_state())
