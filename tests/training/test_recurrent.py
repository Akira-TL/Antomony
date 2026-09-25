from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.environment import SingleAntEnvironment
from mathhackson.training.recurrent import (HIDDEN_WIDTH, MEMORY_LAGS, PARAMETER_LIMITS,
                                            RecurrentPolicy)
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
    assert model.hidden.shape == (HIDDEN_WIDTH,)


def test_sparse_hidden_taps_use_exact_requested_delays():
    assert MEMORY_LAGS == (1, 8, 12, 16)
    model = RecurrentPolicy(4)
    model.phase = "autonomous"
    with torch.no_grad():
        model.input_weights.zero_()
        model.hidden_weights.zero_()
        model.hidden_bias.zero_()
        for index, lag in enumerate(MEMORY_LAGS):
            model.hidden_weights[index, index * HIDDEN_WIDTH + index] = 1.
    model.hidden_history = [torch.zeros(HIDDEN_WIDTH) for _ in range(16)]
    for index, lag in enumerate(MEMORY_LAGS):
        model.hidden_history[-lag][index] = .5
    model.hidden_history[-7][0] = 1.
    model.observe_result(observation())
    assert torch.all(model.hidden[:4] > 0)
    result = model.hidden.clone()
    model.reset_state()
    assert model.hidden_history == []
    model.hidden_history = [torch.zeros(HIDDEN_WIDTH) for _ in range(16)]
    for index, lag in enumerate(MEMORY_LAGS):
        model.hidden_history[-lag][index] = .5
    model.hidden_history[-7][0] = -1.
    model.observe_result(observation())
    assert torch.equal(model.hidden, result)


def test_action_outputs_remain_finite_with_extreme_parameters():
    model = RecurrentPolicy(2)
    model.phase = "autonomous"
    with torch.no_grad():
        model.motor[0, 15] = 1e6
        model.motor[1, 1] = 1e6
    values = observation()
    values[1] = 1.
    action = model.decide(values)
    assert 0. < action.move_probability < 1.
    assert -1. <= action.turn <= 1.
    assert torch.isfinite(action.action_logp)


def test_training_weights_stay_within_group_limits(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    for _ in range(50):
        episode = session.episode
        while session.episode == episode:
            session.step()
    assert all(torch.all(parameter.abs() <= limit) for parameter, limit in
               zip(session.model.parameters, PARAMETER_LIMITS, strict=True))


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


def test_motor_stays_aligned_after_fifty_training_episodes(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    for _ in range(50):
        current = session.episode
        while session.episode == current:
            session.step()
    assert session.motor_alignment() == (6, 6)


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


def test_sensor_bias_changes_observation_but_not_true_goal_or_reward():
    normal, shifted = SingleAntEnvironment(8), SingleAntEnvironment(8)
    shifted.sensor_bias = .35
    assert shifted.relative_food_angle() == normal.relative_food_angle()
    assert not np.array_equal(shifted.observation()[:2], normal.observation()[:2])
    assert shifted.step(False, 0.) == pytest.approx(normal.step(False, 0.))


def test_sensor_task_applies_only_after_eight_steps(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    session.command(RecurrentCommand(action="task", task="sensor"))
    assert session.state().perturbation_kind == "sensor"
    for _ in range(8):
        session.step()
    assert session.env.sensor_bias == 0.
    session.step()
    assert session.env.sensor_bias == session.perturbation


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
    assert all(torch.all(parameter.abs() <= limit) for parameter, limit in
               zip(session.model.parameters, PARAMETER_LIMITS, strict=True))
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


def test_pretrained_motor_cannot_be_reopened_after_phase_change(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    session.command(RecurrentCommand(action="phase", phase="autonomous"))
    with pytest.raises(ValueError, match="基础动作已锁定"):
        session.command(RecurrentCommand(action="phase", phase="motor"))
    assert session.model.phase == "autonomous"
