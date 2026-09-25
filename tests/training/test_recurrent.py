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
