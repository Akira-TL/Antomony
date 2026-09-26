"""可塑方向模块的机械契约，不作为训练有效性的证据。"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution


def test_zero_fast_weights_match_distribution_and_enforce_feedback_order() -> None:
    model = PlasticDirection(DirectionMotor(41), seed=1)
    state = model.initial_state()
    base = torch.tensor([[0.6, 0.8]], requires_grad=True)
    features = torch.ones(1, 45)
    with pytest.raises(ValueError, match="推理"):
        model.feedback(torch.tensor([1.]), state)
    step = model.infer(base, features, state)
    assert torch.equal(step.probabilities, direction_distribution(base.detach()).probs)
    assert torch.equal(step.state.fast, state.fast)
    with pytest.raises(ValueError, match="反馈"):
        model.infer(base, features, step.state)
    result = model.feedback(torch.tensor([1.]), step.state)
    assert not result.boundary
    assert torch.equal(result.state.fast, state.fast)
    with pytest.raises(ValueError, match="推理"):
        model.feedback(torch.tensor([1.]), result.state)


def fixed_outputs(modulation: float = 1., gate: float = 10., *, max_fast: float = .5) -> PlasticDirection:
    model = PlasticDirection(DirectionMotor(41), seed=1, max_fast=max_fast)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([modulation, gate]))
    return model


def advance(model, state, *, reward=1., terminal=False, frozen=False, features=None):
    batch = state.fast.shape[0]
    features = torch.ones(batch, model.feature_width) if features is None else features
    step = model.infer(torch.tensor([[.6, .8]]).repeat(batch, 1), features, state)
    return model.feedback(torch.full((batch,), reward), step.state, terminal=terminal, frozen=frozen)


def test_four_feedback_boundary_short_terminal_and_per_frame_history() -> None:
    model = fixed_outputs()
    state = model.initial_state()
    for frame in range(1, 5):
        result = advance(model, state)
        assert result.boundary == (frame == 4)
        assert len(result.state.history) == frame
        assert bool(result.accepted[0]) == (frame == 4)
        state = result.state
    result = advance(model, state, terminal=True)
    assert result.boundary and len(result.state.rewards) == 0
    assert len(result.state.history) == 5
    assert bool((result.state.fast != state.fast).any())


def test_skip_and_freeze_preserve_already_written_connections_bitwise() -> None:
    model = fixed_outputs()
    learned = advance(model, model.initial_state(), terminal=True).state
    assert bool(learned.fast.any())
    frozen = advance(model, learned, terminal=True, frozen=True)
    assert torch.equal(frozen.state.fast, learned.fast)
    assert frozen.gate_logp is None and not bool(frozen.accepted.any())
    assert len(frozen.state.history) == 2
    with torch.no_grad():
        model.output.bias[1] = -10.
    skipped = advance(model, frozen.state, terminal=True)
    assert torch.equal(skipped.state.fast, learned.fast)
    assert not bool(skipped.accepted.any())
    assert torch.equal(skipped.write_norm, torch.zeros(1))


def test_signed_modulation_opposite_changes_and_both_norm_limits() -> None:
    positive, negative = fixed_outputs(1.), fixed_outputs(-1.)
    plus = advance(positive, positive.initial_state(), terminal=True)
    minus = advance(negative, negative.initial_state(), terminal=True)
    assert torch.equal(plus.state.fast, -minus.state.fast)
    assert 0 < float(plus.write_norm[0].detach()) <= positive.max_step + 1e-6
    model = fixed_outputs(max_fast=.075)
    state = model.initial_state()
    for _ in range(12):
        result = advance(model, state, terminal=True)
        assert bool(torch.isfinite(result.state.fast).all())
        assert float(result.state.fast.detach().norm()) <= .075 + 1e-6
        assert float(result.write_norm[0].detach()) <= .05 + 1e-6
        state = result.state


def test_fast_connections_change_next_distribution_without_changing_base_or_motor() -> None:
    motor = DirectionMotor(41)
    before = [p.detach().clone() for p in motor.parameters()]
    model = fixed_outputs()
    initial = model.initial_state()
    base, features = torch.tensor([[.6, .8]]), torch.ones(1, 45)
    old = model.infer(base, features, initial)
    updated = model.feedback(torch.ones(1), old.state, terminal=True)
    new = model.infer(base, features, updated.state)
    assert not torch.equal(new.probabilities, old.probabilities)
    assert torch.equal(base, torch.tensor([[.6, .8]]))
    assert all(torch.equal(p, v) for p, v in zip(motor.parameters(), before))
    assert all(not p.requires_grad for p in model.motor.parameters())
    assert new.actions[0] == model.motor.decide(new.directions[0].numpy())


def test_batch_and_individual_online_state_and_random_streams_are_separate() -> None:
    model, other = fixed_outputs(), fixed_outputs()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), other.parameters()))
    state, independent = model.initial_state(2), model.initial_state(2)
    features = torch.stack((torch.ones(45), torch.zeros(45)))
    result = advance(model, state, features=features, terminal=True)
    assert bool(result.state.fast[0].any()) and not bool(result.state.fast[1].any())
    assert not bool(independent.fast.any()) and not bool(state.fast.any())
    streams = tuple(torch.Generator().manual_seed(i) for i in (31, 32))
    copies = tuple(torch.Generator().manual_seed(i) for i in (31, 32))
    base = torch.tensor([[.6, .8], [.6, .8]])
    batch = model.infer(base, features, model.initial_state(2), sampled=True, generators=streams)
    single = [model.infer(base[i:i+1], features[i:i+1], model.initial_state(), sampled=True,
                         generators=(copies[i],)) for i in range(2)]
    assert torch.equal(batch.directions, torch.cat([s.directions for s in single]))
    assert all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(streams, copies))
    with pytest.raises(ValueError, match="独立"):
        model.infer(base, features, model.initial_state(2), sampled=True, generators=(streams[0], streams[0]))


def test_recent_and_sparse_fourth_frame_are_distinct_connections() -> None:
    model = fixed_outputs()
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.recent_weights[3].fill_(1.)
        model.sparse_weights[0].fill_(1.)
    state = replace(model.initial_state(), history=(torch.ones(1, 8), torch.zeros(1, 8),
                                                    torch.zeros(1, 8), torch.zeros(1, 8)))
    combined = advance(model, state).state.history[-1]
    combined.sum().backward()
    assert bool(model.recent_weights.grad[3].abs().sum() > 0)
    assert bool(model.sparse_weights.grad[0].abs().sum() > 0)
    assert torch.equal(model.recent_weights.grad[3], model.sparse_weights.grad[0])
    with torch.no_grad():
        model.sparse_weights.zero_()
    recent_only = advance(model, state).state.history[-1]
    assert bool((combined > recent_only).all())
    assert model.recent_weights.data_ptr() != model.sparse_weights.data_ptr()


def test_outer_gradient_crosses_write_but_not_frozen_base_or_motor() -> None:
    model = fixed_outputs(.2)
    base = torch.tensor([[.6, .8]], requires_grad=True)
    features = torch.ones(1, 45, requires_grad=True)
    old = model.infer(base, features, model.initial_state())
    feedback = model.feedback(torch.ones(1), old.state, terminal=True)
    new = model.infer(base, features, feedback.state)
    new.log_probability.sum().backward()
    assert model.output.bias.grad is not None and abs(float(model.output.bias.grad[0])) > 1e-7
    assert model.decay_logit.grad is not None and abs(float(model.decay_logit.grad)) > 1e-7
    assert base.grad is None and features.grad is None
    assert all(p.grad is None for p in model.motor.parameters())


def test_sampled_acceptance_retains_log_probability_for_outer_loss() -> None:
    model = fixed_outputs(.2, 0.)
    step = model.infer(torch.tensor([[.6, .8]]), torch.ones(1, 45), model.initial_state())
    result = model.feedback(torch.ones(1), step.state, terminal=True, sampled_gate=True,
                            generators=(torch.Generator().manual_seed(71),))
    assert result.gate_logp is not None
    (-result.gate_logp.sum()).backward()
    assert abs(float(model.output.bias.grad[1])) == .5


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_observation_feedback_and_state_are_rejected(bad: float) -> None:
    model = fixed_outputs()
    base, features = torch.tensor([[.6, .8]]), torch.ones(1, 45)
    state = model.initial_state()
    invalid = features.clone()
    invalid[0, 0] = bad
    with pytest.raises(ValueError):
        model.infer(base, invalid, state)
    with pytest.raises(ValueError):
        model.infer(base, features[:, :4], state)
    with pytest.raises(ValueError):
        model.infer(base, features, replace(state, fast=torch.full((1, 45, 2), bad)))
    step = model.infer(base, features, state)
    with pytest.raises(ValueError):
        model.feedback(torch.tensor([bad]), step.state)
    assert not bool(state.fast.any())


def test_zero_state_is_bitwise_compatible_for_all_existing_directions() -> None:
    model = fixed_outputs()
    actual = model.infer(DIRECTIONS, torch.ones(16, 45), model.initial_state(16))
    assert torch.equal(actual.probabilities, direction_distribution(DIRECTIONS).probs)


def test_statistics_use_only_received_rewards_and_history_uses_actual_frames() -> None:
    model = fixed_outputs()
    state = model.initial_state()
    for reward in (1., 3., 5., 7.):
        result = advance(model, state, reward=reward, frozen=True)
        state = result.state
    assert torch.allclose(result.statistics[0, :4], torch.tensor([.875, 5 / 6, .8, 5 / 6]))
    for frame in range(5, 18):
        result = advance(model, state, reward=0., frozen=True)
        state = result.state
        assert len(state.history) == min(frame, 16)
    assert len(state.rewards) == 1


def test_feedback_direction_override_is_checked_and_used_without_mutating_inputs() -> None:
    model = fixed_outputs()
    base, features = torch.tensor([[.6, .8]]), torch.ones(1, 45)
    step = model.infer(base, features, model.initial_state())
    features.zero_()
    ordinary = model.feedback(torch.ones(1), step.state, terminal=True)
    reversed_action = model.feedback(torch.ones(1), step.state, terminal=True, executed_direction=DIRECTIONS[12:13])
    assert not torch.equal(ordinary.state.fast, reversed_action.state.fast)
    assert bool(ordinary.state.fast.any())
    with pytest.raises(ValueError, match="16方向"):
        model.feedback(torch.ones(1), step.state, terminal=True, executed_direction=torch.tensor([[.6, .8]]))


def test_nonboundary_and_frozen_feedback_do_not_advance_gate_random_stream() -> None:
    model = fixed_outputs()
    generator = torch.Generator().manual_seed(9)
    initial_random = generator.get_state().clone()
    step = model.infer(torch.tensor([[.6, .8]]), torch.ones(1, 45), model.initial_state())
    short = model.feedback(torch.ones(1), step.state, sampled_gate=True, generators=(generator,))
    assert short.gate_logp is None
    assert torch.equal(generator.get_state(), initial_random)
    model.feedback(torch.ones(1), step.state, terminal=True, frozen=True,
                   sampled_gate=True, generators=(generator,))
    assert torch.equal(generator.get_state(), initial_random)
