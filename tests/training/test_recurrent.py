from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.environment import SingleAntEnvironment
from mathhackson.training.recurrent import RecurrentPolicy
from mathhackson.training.schemas import RecurrentCommand
from mathhackson.training.recurrent_session import RecurrentSession


def observation() -> np.ndarray:
    values = np.zeros(16, np.float32)
    values[15] = 1.
    return values


def test_feedback_changes_next_action_through_hidden_state_only():
    model = RecurrentPolicy(4)
    model.phase = "autonomous"
    with torch.no_grad():
        model.motor.zero_()
        model.input_weights.zero_()
        model.hidden_weights.zero_()
        model.hidden_bias.zero_()
        model.action_weights.zero_()
        model.input_weights[0, 5] = 2.
        model.action_weights[1, 0] = 1.
    before = model.decide(observation())
    positive = observation()
    positive[5] = 1.
    model.observe_result(positive)
    after_positive = model.decide(observation())
    model.reset_state()
    negative = observation()
    negative[5] = -1.
    model.observe_result(negative)
    after_negative = model.decide(observation())
    assert before.turn == 0.
    assert after_positive.turn > 0.
    assert after_negative.turn < 0.
    assert model.hidden.shape == (4,)


def test_memory_phase_keeps_motor_immutable_and_resets_hidden_each_episode(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    for _ in range(40):
        current = session.episode
        while session.episode == current:
            session.step()
        if min(session.motor_alignment()) >= 5:
            break
    assert min(session.motor_alignment()) >= 5
    motor = session.model.motor.detach().clone()
    session.command(RecurrentCommand(action="phase", phase="memory"))
    session.command(RecurrentCommand(action="task", task="mixed"))
    assert not any(any(row) for row in session.state().groups[0].trainable)
    assert all(value == 0. for value in session.state().hidden)
    start_weights = session.model.action_weights.detach().clone()
    for _ in range(8):
        current = session.episode
        while session.episode == current:
            session.step()
        assert all(value == 0. for value in session.state().hidden)
    assert torch.equal(motor, session.model.motor)
    assert not torch.equal(start_weights, session.model.action_weights)
    assert session.model.outer_updates > 0
    assert session.history[-1].checkpoint


def test_shift_is_environment_only_and_starts_after_feedback_window(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    session.command(RecurrentCommand(action="task", task="shift"))
    assert session.perturbation in (-.25, .25)
    before = session.env.observation().copy()
    session.env.turn_bias = session.perturbation
    assert np.array_equal(before, session.env.observation())
    session.env.turn_bias = 0.
    for _ in range(8):
        session.step()
    assert session.env.turn_bias == 0.
    session.step()
    assert session.env.turn_bias == session.perturbation


def test_autonomous_has_state_changes_but_no_parameter_updates(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    session.command(RecurrentCommand(action="phase", phase="autonomous"))
    before = [value.detach().clone() for value in session.model.parameters]
    for _ in range(110):
        session.step()
    assert session.history
    assert session.model.outer_updates == 0
    assert all(torch.equal(old, new) for old, new in zip(before, session.model.parameters, strict=True))


def test_memory_stage_requires_motor_alignment(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    with pytest.raises(ValueError, match="基础动作尚未通过定向检查"):
        session.command(RecurrentCommand(action="phase", phase="memory"))
    assert session.model.phase == "motor"


def test_turn_bias_keeps_physical_turn_within_maximum():
    environment = SingleAntEnvironment(2)
    environment.turn_bias = .25
    environment.step(False, 1.)
    assert environment.heading == pytest.approx(np.pi / 18)


def test_feedback_gate_writes_only_for_next_inference_and_reset_clears_fast_state():
    model = RecurrentPolicy(3)
    model.phase = "autonomous"
    model.write_mode = "learned"
    with torch.no_grad():
        model.motor.zero_()
        model.action_weights.zero_()
        model.gate_weights.zero_()
        model.gate_weights[-1] = 10.
        model.write_weights.zero_()
        model.write_weights[2, -1] = 10.
    before = model.decide(observation())
    motor = model.motor.detach().clone()
    write = model.observe_result(observation())
    after = model.decide(observation())
    assert before.turn == 0.
    assert write.requested and write.wrote
    assert after.turn > 0.
    assert model.fast[2] > 0.
    assert torch.equal(motor, model.motor)
    model.reset_state()
    assert torch.count_nonzero(model.fast) == 0
    assert model.decide(observation()).turn == 0.


def test_feedback_gate_can_skip_and_fast_parameters_remain_bounded():
    model = RecurrentPolicy(3)
    model.phase = "autonomous"
    model.write_mode = "learned"
    with torch.no_grad():
        model.gate_weights.zero_()
        model.gate_weights[-1] = -10.
        model.write_weights.zero_()
        model.write_weights[:, -1] = 10.
    old = model.fast.clone()
    skipped = model.observe_result(observation())
    assert not skipped.requested and not skipped.wrote
    assert torch.equal(old, model.fast)
    assert model.self_updates == 0
    model.write_mode = "always"
    for _ in range(100):
        model.observe_result(observation())
    assert model.self_updates > 0
    assert torch.all(model.fast.abs() <= torch.tensor([.3, .15, .12]))
    assert not model.observe_result(observation(), terminal=True).wrote


def test_adaptive_training_writes_without_changing_pretrained_motor(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    for _ in range(40):
        current = session.episode
        while session.episode == current:
            session.step()
        if min(session.motor_alignment()) >= 5:
            break
    motor = session.model.motor.detach().clone()
    session.command(RecurrentCommand(action="phase", phase="adaptive"))
    session.command(RecurrentCommand(action="task", task="mixed"))
    assert session.state().write_mode == "learned"
    for _ in range(3):
        current = session.episode
        while session.episode == current:
            session.step()
    assert torch.equal(motor, session.model.motor)
    assert session.model.self_updates > 0
    assert all(value == 0. for value in session.state().fast)
    record = session.history[-1]
    with np.load(session.directory / record.checkpoint) as archive:
        assert archive["gates"].shape == (record.steps, 3)
        assert archive["fast_states"].shape == (record.steps, 3)
        assert np.max(np.abs(archive["fast_states"])) <= .3


def test_write_comparison_is_only_available_without_external_training(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    with pytest.raises(ValueError, match="停止外部训练"):
        session.command(RecurrentCommand(action="write_mode", write_mode="always"))
    session.command(RecurrentCommand(action="phase", phase="autonomous"))
    session.command(RecurrentCommand(action="write_mode", write_mode="always"))
    assert session.state().write_mode == "always"
