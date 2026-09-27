"""成对课程和留一回报基线的合成机械测试，不使用正式研究分区。"""
import copy
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
from mathhackson.training.foraging.plastic_course.rollout import (
    CourseTrace, outer_loss, paired_baseline, returns, rollout,
)
from mathhackson.training.foraging.plastic_direction import PlasticDirection


def synthetic_plan(*, batch=4, **options):
    source = run.Source(path="unused-synthetic-model.npz", sha256="a" * 64)
    return run.Protocol(motor=source, updates=2, evaluation_batches=1,
        course=CueConfig(steps=8, batch=batch), initializations=(
            run.Initialization(seed=11, foundation=source, train_seed=100, evaluation_seed=200),
            run.Initialization(seed=12, foundation=source, train_seed=300, evaluation_seed=400)),
        **options)


def synthetic_trace(rewards, *, accepted=None, logp=None, gate_logp=()):
    steps, batch = rewards.shape
    scalar = torch.zeros_like(rewards)
    state = PlasticDirection(DirectionMotor(8), seed=9).initial_state(batch)
    return CourseTrace(rewards=rewards, logp=scalar if logp is None else logp,
        gate_logp=gate_logp, probabilities=torch.zeros(steps, batch, 16),
        directions=torch.zeros(steps, batch, 2),
        accepted=torch.zeros_like(rewards, dtype=torch.bool) if accepted is None else accepted,
        modulation=scalar, gate_probability=scalar, write_norm=scalar,
        fast=torch.zeros(steps, batch, 45, 2), eligibility=torch.zeros(steps, batch, 45, 2),
        hidden=torch.zeros(steps, batch, 8), final_state=state)


def test_old_protocol_defaults_and_pair_requirements():
    old = synthetic_plan().model_dump(exclude={"paired_courses", "baseline_mode"})
    restored = run.Protocol(**old)
    assert restored.paired_courses is False and restored.baseline_mode == "history"
    for options in ({"baseline_mode": "paired"},
                    {"paired_courses": True, "batch": 3},
                    {"paired_courses": True, "batch": 3, "baseline_mode": "paired"},
                    {"baseline_mode": "unknown"}):
        with pytest.raises(ValueError):
            synthetic_plan(**options)
    assert synthetic_plan(batch=3).course.batch == 3
    assert synthetic_plan(paired_courses=True, baseline_mode="paired").course.batch == 4


@pytest.mark.parametrize("paired", [False, True])
def test_complete_courses_are_copied_without_storage_aliases(paired):
    plan = synthetic_plan(paired_courses=paired)
    actual = run.training_episode(71, plan)
    source = cue_episode(71, CueConfig(steps=8, batch=2 if paired else 4))
    assert actual.config == plan.course
    expected_conditions = tuple(condition for condition in source.conditions for _ in range(2)) if paired else source.conditions
    assert actual.conditions == expected_conditions
    for name in ("observations", "targets", "clean_targets", "cue_ids", "reward_flips"):
        original, value = getattr(source, name), getattr(actual, name)
        expected = original.repeat_interleave(2, dim=1) if paired else original
        assert torch.equal(value, expected)
        if paired:
            sibling = value[:, 1].clone()
            value[:, 0] = ~value[:, 0] if value.dtype == torch.bool else value[:, 0] + 1
            assert torch.equal(value[:, 1], sibling)
            assert torch.equal(original, expected[:, ::2])


def test_paired_courses_keep_separate_online_states_and_random_streams():
    episode = run.training_episode(71, synthetic_plan(paired_courses=True))
    model = PlasticDirection(DirectionMotor(8), seed=9, direction_mode="bounded")
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([.2, 0.]))
    base = torch.tensor([.6, .8]).repeat(8, 4, 1)
    together = rollout(model, episode, base, seed=72, training=True)
    assert not torch.equal(together.directions[:, 0], together.directions[:, 1])
    assert not torch.equal(together.accepted[:, 0], together.accepted[:, 1])
    for index in range(4):
        alone = replace(episode, config=CueConfig(steps=8, batch=1),
            conditions=(episode.conditions[index],),
            observations=episode.observations[:, index:index + 1].clone(),
            targets=episode.targets[:, index:index + 1].clone(),
            clean_targets=episode.clean_targets[:, index:index + 1].clone(),
            cue_ids=episode.cue_ids[:, index:index + 1].clone(),
            reward_flips=episode.reward_flips[:, index:index + 1].clone())
        separate = rollout(model, alone, base[:, index:index + 1], seed=72 + 1009 * index, training=True)
        for name in ("directions", "accepted", "rewards"):
            assert torch.equal(getattr(together, name)[:, index:index + 1], getattr(separate, name))
        for name in ("fast", "eligibility", "hidden"):
            assert torch.allclose(getattr(together, name)[:, index:index + 1], getattr(separate, name), atol=1e-6)
    sibling = together.final_state.fast[1].clone()
    with torch.no_grad():
        together.final_state.fast[0].add_(1.)
    assert torch.equal(together.final_state.fast[1], sibling)


def test_paired_baseline_uses_only_other_trajectory_including_cost_and_detaches():
    rewards = torch.tensor([[1., 10., 100., 1000.], [2., 20., 200., 2000.]], requires_grad=True)
    accepted = torch.tensor([[False, True, False, True], [True, False, True, False]])
    trace = synthetic_trace(rewards, accepted=accepted)
    expected = torch.tensor([[18., 1., 1998., 199.], [20., 0., 2000., 198.]])
    actual = paired_baseline(trace, gamma=.5, write_cost=2.)
    assert torch.equal(actual, expected)
    assert not actual.requires_grad and actual.grad_fn is None
    changed = rewards.detach().clone()
    changed[:, 0] += 100.
    altered = paired_baseline(replace(trace, rewards=changed), gamma=.5, write_cost=2.)
    assert torch.equal(altered[:, 0], actual[:, 0])
    assert not torch.equal(altered[:, 1], actual[:, 1])
    assert torch.equal(altered[:, 2:], actual[:, 2:])
    for invalid in (synthetic_trace(torch.zeros(2, 3)),
                    replace(trace, rewards=torch.zeros(2, 0), accepted=torch.zeros(2, 0, dtype=torch.bool))):
        with pytest.raises(ValueError):
            paired_baseline(invalid)


def test_explicit_baseline_gate_credits_only_future_and_never_backpropagates():
    rewards = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    accepted = torch.zeros(8, 2, dtype=torch.bool)
    accepted[3, 0], accepted[7, 1] = True, True
    logp, gate, terminal = (torch.zeros(shape, requires_grad=True) for shape in ((8, 2), (2,), (2,)))
    trace = synthetic_trace(rewards, accepted=accepted, logp=logp, gate_logp=((3, gate), (7, terminal)))
    baseline = paired_baseline(trace, gamma=.5, write_cost=.2).requires_grad_()
    loss, credit = outer_loss(trace, baseline, gamma=.5, write_cost=.2)
    loss.backward()
    all_credit = returns(rewards - .2 * accepted.float(), .5)
    assert torch.equal(credit, all_credit.mean(dim=1))
    assert baseline.grad is None
    assert torch.allclose(logp.grad, -(all_credit - baseline.detach()) / 16)
    assert torch.allclose(gate.grad, -(.5 * all_credit[4] - .2 * accepted[3].float() - .5 * baseline.detach()[4]) / 16)
    assert torch.equal(terminal.grad, .2 * accepted[7].float() / 16)


def test_old_one_dimensional_loss_is_numerically_unchanged():
    rewards = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 17
    accepted = torch.zeros(8, 2, dtype=torch.bool)
    accepted[3::4, 0] = True
    trace = synthetic_trace(rewards, accepted=accepted, logp=rewards * -.3,
        gate_logp=((3, torch.tensor([-.7, -.2])), (7, torch.tensor([-.4, -.8]))))
    baseline = torch.arange(8, dtype=torch.float32) / 20
    credit = returns(rewards - .002 * accepted.float(), .97)
    expected = -(trace.logp * (credit - baseline[:, None])).sum()
    for tick, gate in trace.gate_logp:
        following = .97 * credit[tick + 1] if tick + 1 < 8 else torch.zeros_like(credit[tick])
        value = following - .002 * accepted[tick].float()
        control = .97 * baseline[tick + 1] if tick + 1 < 8 else baseline.new_zeros(())
        expected = expected - (gate * (value - control)).sum()
    actual, actual_credit = outer_loss(trace, baseline)
    assert torch.equal(actual, expected / 16) and torch.equal(actual_credit, credit.mean(dim=1))
    assert torch.equal(outer_loss(trace, baseline[:, None].expand(-1, 2))[0], actual)
    for invalid in (torch.zeros(8, 1), torch.zeros(2, 8), torch.full((8, 2), float("nan"))):
        with pytest.raises(ValueError):
            outer_loss(trace, invalid)


def test_exact_bernoulli_enumeration_preserves_expected_policy_gradient():
    logit = torch.tensor(-.8472978603872037, dtype=torch.float64, requires_grad=True)
    distribution = torch.distributions.Bernoulli(logits=logit)
    expected = [torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)]
    for first in (0., 1.):
        for second in (0., 1.):
            actions = torch.tensor([[first, second]], dtype=torch.float64)
            logp = distribution.log_prob(actions)
            probability = logp.sum().exp().detach()
            trace = synthetic_trace(2 * actions - 1, logp=logp)
            for index, baseline in enumerate((torch.zeros(1, dtype=torch.float64), paired_baseline(trace))):
                loss, _ = outer_loss(trace, baseline)
                gradient = torch.autograd.grad(loss, logit, retain_graph=True)[0]
                expected[index] += probability * gradient
    exact = -2 * distribution.probs.detach() * (1 - distribution.probs.detach())
    assert torch.allclose(expected[0], exact, atol=1e-14, rtol=0)
    assert torch.allclose(expected[1], expected[0], atol=1e-14, rtol=0)


@pytest.mark.parametrize("mode", ["history", "paired"])
def test_training_keeps_step_budget_and_initial_evaluation_unchanged(tmp_path: Path, monkeypatch, mode):
    plan = synthetic_plan(paired_courses=True, baseline_mode=mode, evaluate_initial=True)
    model = PlasticDirection(DirectionMotor(8), seed=9)
    base = MemoryPolicy(FeedforwardPolicy(10))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    episodes, traces, baselines = [], [], []
    original_episode, original_rollout, original_loss = run.cue_episode, run.rollout, run.outer_loss

    class NoUpdates:
        def __init__(self, *args, **kwargs):
            pass

        def zero_grad(self):
            model.zero_grad(set_to_none=True)

        def step(self):
            pass

    def observed_episode(seed, config, **options):
        episodes.append((seed, config.batch, options))
        return original_episode(seed, config, **options)

    def observed_rollout(*args, **options):
        trace = original_rollout(*args, **options)
        traces.append(trace)
        return trace

    def observed_loss(trace, baseline, **options):
        baselines.append(baseline.clone())
        return original_loss(trace, baseline, **options)

    monkeypatch.setattr(run.torch.optim, "AdamW", NoUpdates)
    monkeypatch.setattr(run, "cue_episode", observed_episode)
    monkeypatch.setattr(run, "rollout", observed_rollout)
    monkeypatch.setattr(run, "outer_loss", observed_loss)
    monkeypatch.setattr(run, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "save_trace", lambda *args, **kwargs: None)
    run.train(model, base, plan.initializations[0], plan, tmp_path, time.monotonic() + 30)
    assert episodes == [(101, 2, {}), (102, 2, {})]
    assert sum(trace.rewards.numel() for trace in traces) == 8 * 4 * 2
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    if mode == "paired":
        for trace, actual in zip(traces, baselines):
            assert torch.equal(actual, paired_baseline(trace, gamma=plan.gamma, write_cost=plan.write_cost))
    else:
        assert torch.equal(baselines[0], torch.zeros(8))
        prior = returns(traces[0].rewards - plan.write_cost * traces[0].accepted.float(), plan.gamma).mean(dim=1)
        assert torch.equal(baselines[1], (1 - plan.baseline_decay) * prior)
    episodes.clear()
    initial = copy.deepcopy(model)
    run.evaluate(initial, base, plan.initializations[0], plan, tmp_path,
        time.monotonic() + 30, structure_update=0)
    assert episodes == [(200, 4, {"condition": condition}) for condition in ("reference", "association", "transient")]
    assert model.training and any(parameter.requires_grad for parameter in model.parameters())
    assert not initial.training and not any(parameter.requires_grad for parameter in initial.parameters())
