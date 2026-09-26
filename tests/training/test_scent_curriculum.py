from pathlib import Path

import numpy as np
import torch

from mathhackson.training.roundtrip_policy import RoundTripPolicy
from mathhackson.training.scent_curriculum import (_memory_examples,
                                                   _memory_turns,
                                                   _teacher_trajectory,
                                                   evaluate_scent_memory,
                                                   evaluate_scent_reader,
                                                   evaluate_teacher_scent,
                                                   pretrain_scent_memory,
                                                   pretrain_scent_reader,
                                                   pretrain_teacher_scent)

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


def test_memory_examples_have_identical_current_inputs_with_opposite_targets():
    observations, targets, queries = _memory_examples(np.random.default_rng(4), 32, 20)
    assert torch.equal(observations[1:, :16], observations[1:, 16:])
    assert torch.all(torch.sign(targets[queries].reshape(4, 2, 16)[:, 0]) !=
                     torch.sign(targets[queries].reshape(4, 2, 16)[:, 1]))
    assert not torch.equal(observations[0, :16], observations[0, 16:])


def test_memory_training_forward_matches_action_then_feedback_timing():
    model = RoundTripPolicy(91, FOUNDATION)
    observations, _, _ = _memory_examples(np.random.default_rng(4), 2, 20)
    predicted = torch.tanh(3. * _memory_turns(model, observations[:, :1])).detach().flatten()
    model.phase = "autonomous"
    actual = []
    for values in observations[:, 0].numpy():
        actual.append(model.decide(values).turn)
        model.observe_result(values)
    np.testing.assert_allclose(predicted.numpy(), actual, atol=1e-6)


def test_memory_curriculum_requires_recurrence_and_keeps_motor_frozen():
    model = RoundTripPolicy(91, FOUNDATION)
    pretrain_scent_reader(model)
    before = evaluate_scent_memory(model)
    motor = model.motor.detach().clone()
    release = model.release_weights.detach().clone()
    hidden = model.hidden_weights.detach().clone()
    saved = []
    after = pretrain_scent_memory(model, steps=100, checkpoint_every=50,
                                  checkpoint=lambda step, metrics: saved.append((step, metrics)))
    assert before.full.direction_accuracy < .65
    assert after.full.direction_accuracy > .95
    assert after.without_memory.direction_accuracy == .5
    assert after.full.turn_error < before.full.turn_error - .5
    assert [step for step, _ in saved] == [50, 100]
    assert torch.equal(model.motor, motor)
    assert torch.equal(model.release_weights, release)
    assert not torch.equal(model.hidden_weights, hidden)


def test_teacher_trajectory_uses_local_observations_and_real_feedback():
    trajectory = _teacher_trajectory(1000)
    carrying = trajectory.observations[:, 16] > .5
    assert carrying.any()
    assert np.all(trajectory.observations[carrying, :3] == 0.)
    assert np.all(trajectory.feedback[carrying, :3] == 0.)
    assert np.any(np.linalg.norm(trajectory.observations[carrying, 9:11], axis=1) > .005)
    assert not np.array_equal(trajectory.observations, trajectory.feedback)


def test_teacher_forward_matches_live_action_and_feedback():
    model = RoundTripPolicy(91, FOUNDATION)
    trajectory = _teacher_trajectory(1000)
    observations = torch.from_numpy(trajectory.observations[:, None, :])
    feedback = torch.from_numpy(trajectory.feedback[:, None, :])
    predicted = torch.tanh(3. * _memory_turns(model, observations, feedback=feedback)).detach().flatten()
    model.phase = "autonomous"
    actual = []
    for before, after in zip(trajectory.observations, trajectory.feedback, strict=True):
        actual.append(model.decide(before).turn)
        model.observe_result(after)
    np.testing.assert_allclose(predicted.numpy(), actual, atol=1e-6)


def test_teacher_curriculum_reduces_heldout_return_error_without_unfreezing_motor():
    model = RoundTripPolicy(91, FOUNDATION)
    pretrain_scent_reader(model)
    before = evaluate_teacher_scent(model)
    motor = model.motor.detach().clone()
    release = model.release_weights.detach().clone()
    saved = []
    after = pretrain_teacher_scent(model, steps=50, checkpoint_every=25,
                                   checkpoint=lambda step, metrics: saved.append((step, metrics)))
    assert before.return_error > .7
    assert after.return_error < before.return_error - .3
    assert [step for step, _ in saved] == [25, 50]
    assert torch.equal(model.motor, motor)
    assert torch.equal(model.release_weights, release)
