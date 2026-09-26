"""只验证概率映射的数学边界，不执行学习课程或推断效果。"""
from dataclasses import replace

import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution


@pytest.mark.parametrize("mode", ["unit", "bounded"])
@pytest.mark.parametrize("scale", [1., 1.000008, .999992])
def test_zero_residual_keeps_every_legal_base_probability_bitwise(mode, scale):
    model = PlasticDirection(DirectionMotor(8), direction_mode=mode)
    base = DIRECTIONS * scale
    step = model.infer(base, torch.ones(16, 45), model.initial_state(16))
    assert torch.equal(step.probabilities, direction_distribution(base).probs)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 155


def test_bounded_radial_halving_reduces_preference_while_unit_is_unchanged():
    base, features = torch.tensor([[1., 0.]]), torch.ones(1, 1)
    legacy = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2.)
    bounded = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2., direction_mode="bounded")
    state = replace(bounded.initial_state(), fast=torch.tensor([[[-.5, 0.]]]))
    actual = bounded.infer(base, features, state)
    original = legacy.infer(base, features, state)
    assert legacy.direction_mode == "unit"
    assert torch.equal(original.probabilities, direction_distribution(base).probs)
    assert torch.equal(actual.probabilities, direction_distribution(base * .5).probs)
    assert actual.probabilities[0, 0] < original.probabilities[0, 0]
    assert torch.all(actual.probabilities >= .2 / 16)


def test_bounded_cancellation_is_uniform_and_motor_still_receives_unit_directions():
    model = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2., direction_mode="bounded")
    state = replace(model.initial_state(), fast=torch.tensor([[[-1., 0.]]]))
    step = model.infer(torch.tensor([[1., 0.]]), torch.ones(1, 1), state, sampled=True,
        generators=(torch.Generator().manual_seed(12),))
    assert torch.equal(step.probabilities, torch.full((1, 16), 1 / 16))
    assert torch.allclose(step.directions.norm(dim=1), torch.ones(1))
    assert any(torch.equal(step.directions[0], item) for item in DIRECTIONS)
    assert step.actions[0] == model.motor.decide(step.directions[0].numpy())


def test_bounded_outside_unit_ball_matches_unit_mode_bitwise():
    base, features = torch.tensor([[.6, .8]]), torch.ones(1, 1)
    unit = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2.)
    bounded = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2., direction_mode="bounded")
    state = replace(unit.initial_state(), fast=torch.tensor([[[.7, .2]]]))
    assert (base + state.fast[:, 0]).norm() > 1
    assert torch.equal(unit.infer(base, features, state).probabilities,
        bounded.infer(base, features, state).probabilities)


@pytest.mark.parametrize("scale", [1., 1.000008, .999992])
def test_bounded_zero_residual_keeps_radial_gradient_at_initial_boundary(scale):
    model = PlasticDirection(DirectionMotor(8), feature_width=1, direction_mode="bounded")
    fast = torch.zeros(1, 1, 2, requires_grad=True)
    base = torch.tensor([[scale, 0.]], requires_grad=True)
    state = replace(model.initial_state(), fast=fast)
    step = model.infer(base, torch.ones(1, 1), state)
    step.probabilities[0, 0].backward()
    assert torch.isfinite(fast.grad).all() and fast.grad[0, 0, 0] > .1
    assert base.grad is None
    assert all(parameter.grad is None for parameter in model.motor.parameters())


def test_bounded_nonzero_residual_on_boundary_keeps_radial_gradient():
    model = PlasticDirection(DirectionMotor(8), feature_width=1, max_fast=2., direction_mode="bounded")
    fast = torch.tensor([[[-1., 1.]]], requires_grad=True)
    state = replace(model.initial_state(), fast=fast)
    step = model.infer(torch.tensor([[1., 0.]]), torch.ones(1, 1), state)
    step.probabilities[0, 4].backward()
    assert torch.isfinite(fast.grad).all() and fast.grad[0, 0, 1] > .1


@pytest.mark.parametrize("mode", ["other", "", None])
def test_unknown_direction_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="方向模式"):
        PlasticDirection(DirectionMotor(8), direction_mode=mode)
