from pathlib import Path

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
    assert decision.turn == pytest.approx(float(torch.tanh(output[1])))
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


@pytest.mark.parametrize("phase", ["motor", "meta", "autonomous"])
def test_all_frozen_stays_bitwise_identical_across_episodes(tmp_path: Path, phase):
    session = TrainingSession(tmp_path)
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
    session.command(Command(action="phase", phase="meta"))
    session.command(Command(action="freeze", group="gate", frozen=True))
    before = session.model.weights.detach().clone()
    for _ in range(100):
        session.step()
    assert torch.equal(before[2], session.model.weights[2])
    assert not torch.equal(before[0:2], session.model.weights[0:2])
    assert session.model.outer_updates > 0 and session.model.self_updates > 0


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
