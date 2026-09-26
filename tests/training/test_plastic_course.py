from dataclasses import replace

import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.feedback_cues import CueConfig, cue_episode
from mathhackson.training.foraging.plastic_course.rollout import matched_schedule, outer_loss, returns, rollout
from mathhackson.training.foraging.plastic_direction import PlasticDirection


def sample(training=False, mode="learned"):
    episode = cue_episode(908001, CueConfig(steps=8, batch=2))
    model = PlasticDirection(DirectionMotor(13), seed=19, max_step=.2, max_fast=2.)
    base = torch.zeros(8, 2, 2)
    base[..., 0] = 1.
    return model, episode, base, rollout(model, episode, base, seed=908002, training=training, mode=mode)


def test_matched_schedule_preserves_per_individual_count_and_boundary():
    accepted = torch.zeros(24, 3, dtype=torch.bool)
    accepted[3::8, 0] = True
    accepted[3::4, 1] = True
    matched = matched_schedule(accepted, 912)
    assert torch.equal(matched.sum(dim=0), accepted.sum(dim=0))
    assert not matched[torch.arange(24) % 4 != 3].any()
    assert torch.equal(matched, matched_schedule(accepted, 912))
    with pytest.raises(ValueError):
        matched_schedule(accepted.to(torch.float32), 912)


def test_gate_credit_excludes_same_action_reward():
    _, _, _, trace = sample(training=True)
    gate = torch.tensor([.2, .3], requires_grad=True)
    trace = replace(trace, logp=torch.zeros_like(trace.logp), gate_logp=((3, gate),))
    changed = trace.rewards.clone()
    changed[3] += 10.
    first, _ = outer_loss(trace, torch.zeros(8), write_cost=0)
    second, _ = outer_loss(replace(trace, rewards=changed), torch.zeros(8), write_cost=0)
    assert torch.equal(first, second)
    changed[4] += 10.
    third, _ = outer_loss(replace(trace, rewards=changed), torch.zeros(8), write_cost=0)
    assert not torch.equal(first, third)
    assert torch.equal(returns(torch.tensor([[1.], [2.], [3.]]), 1.), torch.tensor([[6.], [5.], [3.]]))


def test_outer_training_only_reaches_modulation_not_motor_or_frozen_base():
    model, _, _, trace = sample(training=True)
    loss, credit = outer_loss(trace, torch.zeros(8))
    loss.backward()
    assert torch.isfinite(loss)
    assert credit.shape == (8,)
    assert model.output.weight.grad is not None and model.output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.motor.parameters())
    assert trace.fast.shape == (8, 2, 45, 2)
    assert trace.hidden.shape == (8, 2, 8)


def test_off_and_always_are_real_write_interventions_with_common_action_stream():
    model, episode, base, off = sample(mode="off")
    always = rollout(model, episode, base, seed=908002, mode="always")
    assert not off.fast.any() and not off.accepted.any()
    assert always.accepted.sum() == 4
    assert torch.equal(off.directions[:4], always.directions[:4])
    assert torch.equal(off.probabilities[:4], always.probabilities[:4])
    schedule = matched_schedule(always.accepted, 29)
    matched = rollout(model, episode, base, seed=908002, mode="matched", schedule=schedule)
    assert torch.equal(always.fast, matched.fast)
    assert torch.equal(always.rewards, matched.rewards)
