"""接受判断与外层归因的合成机械契约，不读取正式课程记录。"""
from dataclasses import replace
from pathlib import Path
import time

import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.feedback_cues import CueConfig, cue_episode
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.plastic_course import run
from mathhackson.training.foraging.plastic_course.rollout import outer_loss, rollout, streams
from mathhackson.training.foraging.plastic_direction import PlasticDirection


def synthetic_trace(*, modulation=.2, gate=0., training=False, batch=2):
    model = PlasticDirection(DirectionMotor(8), seed=9, max_step=.2, max_fast=2., direction_mode="bounded")
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([modulation, gate]))
    episode = cue_episode(71, CueConfig(steps=8, batch=batch))
    base = torch.tensor([.6, .8]).repeat(8, batch, 1).requires_grad_()
    return model, episode, base, rollout(model, episode, base, seed=72, training=training)


def test_training_samples_each_individual_gate_while_evaluation_uses_threshold():
    model, episode, base, sampled = synthetic_trace(modulation=0., training=True, batch=8)
    generators = streams(72 + 100_000_000, 8)
    expected = torch.stack([torch.stack([
        torch.bernoulli(torch.tensor(.5), generator=generator).bool() for generator in generators
    ]) for _ in range(2)])
    assert expected.any() and not expected.all()
    assert torch.equal(sampled.accepted[3::4], expected)
    assert not sampled.accepted[torch.arange(8) % 4 != 3].any()
    assert [tick for tick, _ in sampled.gate_logp] == [3, 7]
    assert all(logp.requires_grad for _, logp in sampled.gate_logp)
    evaluated = rollout(model, episode, base, seed=72)
    assert evaluated.accepted[3::4].all()
    assert evaluated.gate_logp == ()
    assert torch.equal(sampled.directions, evaluated.directions)
    assert len(evaluated.directions.reshape(-1, 2).unique(dim=0)) > 1


@pytest.mark.parametrize("logit, accepted", [(-.01, False), (0., True), (.01, True)])
def test_evaluation_accepts_probability_equal_to_half(logit: float, accepted: bool):
    _, _, _, trace = synthetic_trace(gate=logit)
    assert torch.equal(trace.accepted[3::4], torch.full((2, 2), accepted))
    assert trace.gate_logp == ()


@pytest.mark.parametrize("accepted", [False, True])
def test_terminal_gate_counts_only_acceptance_cost_even_when_write_is_zero(accepted: bool):
    _, _, _, trace = synthetic_trace(modulation=0., gate=30. if accepted else -30.)
    assert not trace.write_norm.any() and not trace.fast.any()
    assert bool(trace.accepted[-1].all()) == accepted
    logits = torch.zeros(2, requires_grad=True)
    logp = torch.distributions.Bernoulli(logits=logits).log_prob(trace.accepted[-1].float())
    isolated = replace(trace, logp=torch.zeros_like(trace.logp), gate_logp=((7, logp),))
    loss, _ = outer_loss(isolated, torch.ones(8), write_cost=.2)
    changed = trace.rewards.clone()
    changed[-1] += 99.
    altered_loss, _ = outer_loss(replace(isolated, rewards=changed), torch.ones(8), write_cost=.2)
    assert torch.equal(loss, altered_loss)
    expected_loss = .2 * logp.sum() / trace.rewards.numel() if accepted else logp.sum() * 0
    assert torch.allclose(loss, expected_loss)
    gradient = torch.autograd.grad(loss, logits)[0]
    expected_gradient = .2 * .5 / trace.rewards.numel() if accepted else 0.
    assert torch.allclose(gradient, torch.full_like(logits, expected_gradient))


def test_log_copies_are_detached_but_next_action_keeps_gradient_through_write():
    model, _, base, trace = synthetic_trace(gate=30., training=True)
    assert trace.accepted[3].all() and (trace.write_norm[3] > 0).all()
    for logged in (trace.probabilities, trace.directions, trace.accepted, trace.modulation,
                   trace.gate_probability, trace.write_norm, trace.fast, trace.eligibility, trace.hidden):
        assert not logged.requires_grad and logged.grad_fn is None
    assert trace.final_state.fast.requires_grad and trace.final_state.history[-1].requires_grad
    trace.logp[4].sum().backward()
    assert model.output.bias.grad is not None and torch.isfinite(model.output.bias.grad).all()
    assert abs(float(model.output.bias.grad[0])) > 1e-7
    assert model.output.bias.grad[1] == 0
    assert base.grad is None and all(parameter.grad is None for parameter in model.motor.parameters())


def test_training_uses_previous_batches_baseline_before_current_credit(tmp_path: Path, monkeypatch):
    model, episode, base, trace = synthetic_trace(gate=30.)
    source = run.Source(path="unused-synthetic-model.npz", sha256="a" * 64)
    plan = run.Protocol(motor=source, updates=2, course=episode.config, baseline_decay=.75,
        initializations=(
            run.Initialization(seed=11, foundation=source, train_seed=100, evaluation_seed=200),
            run.Initialization(seed=12, foundation=source, train_seed=300, evaluation_seed=400)))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    seen: list[torch.Tensor] = []
    events: list[str] = []

    class NoUpdates:
        def __init__(self, *args, **kwargs):
            pass

        def zero_grad(self):
            model.zero_grad(set_to_none=True)

        def step(self):
            events.append("step")

    def synthetic_loss(actual_trace, baseline, **options):
        events.append("loss")
        seen.append(baseline.clone())
        assert actual_trace is trace and not baseline.requires_grad
        return model.decay_logit * 0, torch.full((8,), 2. if len(seen) == 1 else 100.)

    monkeypatch.setattr(run.torch.optim, "AdamW", NoUpdates)
    monkeypatch.setattr(run, "cue_episode", lambda *args: episode)
    monkeypatch.setattr(run, "base_directions", lambda *args: base)
    monkeypatch.setattr(run, "rollout", lambda *args, **kwargs: trace)
    monkeypatch.setattr(run, "outer_loss", synthetic_loss)
    monkeypatch.setattr(run, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "save_trace", lambda *args, **kwargs: None)
    run.train(model, MemoryPolicy(FeedforwardPolicy(10)), plan.initializations[0], plan,
        tmp_path, time.monotonic() + 30)
    assert events == ["loss", "step", "loss", "step"]
    assert torch.equal(seen[0], torch.zeros(8))
    assert torch.equal(seen[1], torch.full((8,), .5))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
