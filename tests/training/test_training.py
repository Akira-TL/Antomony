from pathlib import Path
import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from mathhackson.training.environment import SingleAntEnvironment
from mathhackson.training.model import GROUPS, SelfModifyingPolicy
from mathhackson.training.schemas import Command
from mathhackson.training.session import TrainingSession


def observation() -> np.ndarray:
    return np.asarray([1., .2, .4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., .1, 1.], np.float32)


def force_gate(model: SelfModifyingPolicy, value: float) -> None:
    with torch.no_grad():
        model.base[2].zero_()
        model.base[2, -1] = value


def test_self_write_is_after_action_and_control_can_modify_itself():
    model = SelfModifyingPolicy(7)
    model.phase = "autonomous"
    force_gate(model, 4.)
    before = model.weights.detach().clone()
    output = before @ torch.from_numpy(observation())
    decision = model.decide(observation())
    assert decision.move == bool(output[0] >= 0.)
    assert decision.turn == pytest.approx(float(torch.tanh(3. * output[1])))
    assert decision.wrote
    assert not torch.equal(before[2], model.weights[2])
    assert model.outer_updates == 0


def test_gate_can_skip_without_changing_any_weights():
    model = SelfModifyingPolicy(7)
    model.phase = "autonomous"
    force_gate(model, -4.)
    before = model.weights.detach().clone()
    assert not model.decide(observation()).requested_write
    assert torch.equal(before, model.weights)
    assert model.self_updates == 0


def test_basic_training_learns_both_turn_signs_without_runtime_direction_rule():
    left, right = SelfModifyingPolicy(7), SelfModifyingPolicy(7)
    left_input, right_input = observation(), observation()
    left_input[0:2] = [0., 1.]
    right_input[0:2] = [0., -1.]
    for _ in range(18):
        for model, current in ((left, left_input), (right, right_input)):
            decisions = [model.decide(current) for _ in range(12)]
            model.finish(decisions, [0.] * len(decisions), 0., True)
    left.phase = right.phase = "autonomous"
    assert left.decide(left_input).turn > .5
    assert right.decide(right_input).turn < -.5
    assert left.outer_updates == right.outer_updates == 18


@pytest.mark.parametrize("phase", ["motor", "meta", "autonomous"])
def test_all_frozen_stays_bitwise_identical_across_episodes(tmp_path: Path, phase):
    session = TrainingSession(tmp_path)
    if phase == "meta":
        session.model.phase = "meta"
    else:
        session.command(Command(action="phase", phase=phase))
    session.command(Command(action="freeze_all", frozen=True))
    before = session.model.weights.detach().clone()
    for _ in range(110):
        session.step()
    assert session.history
    assert torch.equal(before, session.model.weights)
    assert session.model.outer_updates == session.model.self_updates == 0


def test_partial_freeze_masks_both_update_paths(tmp_path: Path):
    session = TrainingSession(tmp_path)
    session.model.phase = "meta"
    session.command(Command(action="freeze", group="gate", frozen=True))
    before = session.model.weights.detach().clone()
    for _ in range(100):
        session.step()
    assert torch.equal(before[2], session.model.weights[2])
    assert torch.equal(before[0:2], session.model.weights[0:2])
    assert session.model.outer_updates == 0 and session.model.self_updates > 0


def test_basic_stage_only_trains_action_rows_and_records_true_values(tmp_path: Path):
    session = TrainingSession(tmp_path)
    before = session.model.weights.detach().clone()
    while not session.history:
        session.step()
    assert torch.equal(before[2:], session.model.weights[2:])
    assert not torch.equal(before[:2], session.model.weights[:2])
    record = session.history[-1]
    with np.load(session.directory / record.checkpoint) as archive:
        assert np.array_equal(archive["start"], before.numpy())
        assert np.array_equal(archive["end"], session.model.weights.detach().numpy())
        assert len(archive["weights"]) == record.steps
        assert archive["rewards"].sum() == pytest.approx(record.reward)
    assert session.model.self_updates == 0


def test_motor_stage_has_four_active_action_connections(tmp_path: Path):
    session = TrainingSession(tmp_path)
    mask = session.model.mask().numpy()
    expected = np.zeros((2, 16), np.float32)
    expected[0, [0, 2, 15]] = 1.
    expected[1, 1] = 1.
    assert np.array_equal(mask[:2], expected)
    assert np.array_equal(session.model.initial[:2].numpy() * (1 - expected), np.zeros((2, 16)))
    while session.episode <= 30:
        session.step()
    weights = session.model.weights.detach().numpy()
    assert np.array_equal(weights[:2] * (1 - expected), np.zeros((2, 16)))
    assert weights[1, 1] != session.model.initial[1, 1]
    assert min(session.motor_alignment()) >= 5
    session.command(Command(action="phase", phase="meta"))
    for _ in range(10):
        current_episode = session.episode
        while session.episode == current_episode:
            session.step()
    assert min(session.motor_alignment()) >= 5


def test_configuration_discards_external_update_not_history(tmp_path: Path):
    session = TrainingSession(tmp_path)
    for _ in range(5):
        session.step()
    before = session.model.weights.detach().clone()
    session.command(Command(action="phase", phase="autonomous"))
    assert session.paused
    assert session.history[-1].steps == 5
    assert session.history[-1].outer_updates == 0
    assert torch.equal(before, session.model.weights)
    for _ in range(100):
        session.step()
    assert session.model.outer_updates == 0


def test_sessions_are_independent_and_pause_step_is_explicit(tmp_path: Path):
    one, two = TrainingSession(tmp_path), TrainingSession(tmp_path)
    assert one.paused and two.paused
    one.command(Command(action="step"))
    assert one.tick == 1 and one.paused
    assert two.tick == 0 and two.paused
    assert one.directory != two.directory
    one.model.manual_frozen.add("action")
    assert not two.model.manual_frozen


def test_environment_follows_boolean_move_and_bounded_turn():
    env = SingleAntEnvironment(1)
    env.step(False, 1.)
    assert np.array_equal(env.position, np.zeros(2))
    assert env.heading == pytest.approx(np.pi / 18)
    env.step(True, 0.)
    assert np.linalg.norm(env.position) == pytest.approx(.18)


def test_random_food_is_default_and_covers_all_directions():
    env = SingleAntEnvironment(4)
    assert env.lesson == "random"
    locations = []
    for _ in range(80):
        locations.append(env.target.copy())
        env.reset()
    assert len({tuple(point) for point in locations}) == len(locations)
    assert all(2.5 <= np.linalg.norm(point) <= 6.5 for point in locations)
    assert {tuple(np.sign(point).astype(int)) for point in locations} == {
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    }
    assert any(point[0] < -2.5 for point in locations)


def test_turning_toward_random_food_has_better_feedback_than_away():
    toward, away = SingleAntEnvironment(1), SingleAntEnvironment(1)
    for env in (toward, away):
        env.target = np.asarray([0., 3.], np.float32)
        env.distance = 3.
    assert toward.step(False, 1.) > away.step(False, -1.)
    assert np.array_equal(toward.position, away.position)
    assert toward.reward > -.01 > away.reward
    assert toward.heading == pytest.approx(math.radians(10))


def test_reject_nonfinite_observations_and_invalid_commands():
    with pytest.raises(ValueError):
        SelfModifyingPolicy(1).decide(np.full(16, np.nan))
    for payload in ({"action": "turn", "max_turn": float("nan")},
                    {"action": "speed", "speed": 1000}, {"action": "freeze", "group": "unknown"}):
        with pytest.raises(ValidationError):
            Command.model_validate(payload)


def test_state_exposes_every_weight_and_effective_freeze(tmp_path: Path):
    session = TrainingSession(tmp_path)
    state = session.state()
    assert len(state.weights) == 39 and all(len(row) == 16 for row in state.weights)
    assert len(state.inputs) == 16
    assert [group.frozen for group in state.groups] == [False, True, True, True, True]
    assert sum(end - start for _, _, start, end in GROUPS) == len(state.weights)
    assert sum(sum(row) for row in state.trainable) == 4
    assert state.trainable[0][0] and not state.trainable[0][1]
    assert state.trainable[1][1] and not state.trainable[1][15]
    session.command(Command(action="freeze", group="action", frozen=True))
    assert not any(any(row) for row in session.state().trainable)
    session.command(Command(action="freeze", group="action", frozen=False))
    session.model.phase = "meta"
    assert sum(sum(row) for row in session.state().trainable) == 592
    assert session.state().groups[0].reason == "预训练动作锁定"


def test_meta_training_keeps_pretrained_motor_weights_even_when_writer_runs(tmp_path: Path):
    session = TrainingSession(tmp_path)
    session.model.phase = "meta"
    before = session.model.weights[:2].detach().clone()
    for _ in range(150):
        session.step()
    assert session.model.self_updates > 0
    assert session.model.outer_updates > 0
    assert torch.equal(before, session.model.weights[:2])


def test_weight_trace_records_actual_steps_and_outer_update(tmp_path: Path):
    session = TrainingSession(tmp_path)
    while not session.history:
        session.step()
    trace = session.weight_trace(0, 0)
    assert len(trace.points) == session.history[0].steps + 2
    assert trace.points[0].source == "initial"
    assert all(point.source == "skip" and point.delta == 0 for point in trace.points[1:-1])
    assert trace.points[-1].source == "outer"
    assert trace.points[-1].changed > 0
    assert trace.points[-1].value == pytest.approx(session.state().weights[0][0])
    assert trace.points[-1].value - trace.points[-2].value == pytest.approx(trace.points[-1].delta)
    assert [point.sequence for point in trace.points] == list(range(len(trace.points)))


def test_weight_trace_captures_self_writes_and_freeze_as_flatline(tmp_path: Path):
    session = TrainingSession(tmp_path)
    session.model.phase = "meta"
    for _ in range(32):
        session.step()
    trace = session.weight_trace(2, 0)
    assert any(point.source == "self" and point.changed > 0 for point in trace.points)
    for previous, current in zip(trace.points, trace.points[1:]):
        if current.source in {"self", "skip"}:
            assert current.value - previous.value == pytest.approx(current.delta, abs=1e-7)
    session.command(Command(action="freeze_all", frozen=True))
    frozen = session.weight_trace(2, 0).points[-1].value
    for _ in range(110):
        session.step()
    assert all(point.value == frozen and point.delta == 0 for point in session.weight_trace(2, 0).points[-110:])
    with pytest.raises(ValueError):
        session.weight_trace(39, 0)


def test_meta_requires_aligned_motor_policy_before_switch(tmp_path: Path):
    session = TrainingSession(tmp_path)
    before = session.state()
    with pytest.raises(ValueError, match="基础动作尚未通过定向检查"):
        session.command(Command(action="phase", phase="meta"))
    assert session.state().phase == "motor"
    assert session.tick == before.tick and not session.history
    assert session.motor_alignment() == (0, 0)
    session.command(Command(action="phase", phase="autonomous"))
    with pytest.raises(ValueError, match="基础动作尚未通过定向检查"):
        session.command(Command(action="phase", phase="meta"))
    assert session.state().phase == "autonomous"
    session.command(Command(action="phase", phase="motor"))
    for _ in range(60):
        current_episode = session.episode
        while session.episode == current_episode:
            session.step()
        left, right = session.motor_alignment()
        if left >= 5 and right >= 5:
            break
    assert left >= 5 and right >= 5
    session.command(Command(action="phase", phase="meta"))
    assert session.state().phase == "meta"
