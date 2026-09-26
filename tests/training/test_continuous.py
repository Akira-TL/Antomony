import gzip

import pytest
import torch
from pydantic import ValidationError

from mathhackson.training.comparison.continuous import ARMS, Condition, ContinuousPlan, Frame, make_actors, run_world
from mathhackson.training.comparison.online_actor import OnlineForager, UpdateRecord
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner
from mathhackson.training.foraging.update_decision import UpdateDecision


@pytest.fixture
def plan(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for i in range(2):
        MemoryPolicy(FeedforwardPolicy(81 + i)).save(sources / f"episode-0000-ant-{i:02d}.npz", update=0, phase="frozen")
        FeedforwardPolicy(81 + i, hidden_width=17).save(sources / f"seed-{81+i}-signal-002400.npz", update=2400, phase="signal")
        (sources / f"ant-{i:02d}").mkdir()
        decision = UpdateDecision()
        decision.updates = 200
        with torch.no_grad():
            decision.value.bias.fill_(1.)
        decision.save(sources / f"ant-{i:02d}" / "step-0200.npz")
    save_motor(sources / "motor.npz", DirectionMotor(41), MotorSnapshot(seed=41, updates=0))
    return ContinuousPlan(seeds=(18999,), environment=ColonyConfig(ants=2, horizon=20, stock=48), checkpoint_every=8,
        adaptation=TrustConfig(window=4, feedback_mode="observed-window"),
        conditions=(Condition(name="reference", disturbance=DisturbanceConfig(signal_strength=0., injury_per_step=0.)),),
        policy_directory=str(sources), gate_directory=str(sources), mlp_directory=str(sources), motor=str(sources / "motor.npz"))


def frames(directory):
    with gzip.open(directory / "trajectory.jsonl.gz", "rt") as stream:
        return [Frame.model_validate_json(line) for line in stream]


def test_reference_three_online_modes_have_identical_complete_trajectories(plan, tmp_path, monkeypatch):
    import mathhackson.training.foraging.candidate_value as branch
    def forbidden(*args, **kwargs):
        raise AssertionError("部署调用未来分支")
    monkeypatch.setattr(branch, "evaluate_candidate", forbidden)
    trajectories = []
    for arm in ARMS:
        directory = tmp_path / arm
        result = run_world(plan, 18999, plan.conditions[0], arm, directory)
        assert result.steps == 20 and result.snapshots == [0, 8, 16, 20]
        assert result.writes == [0, 0]
        assert len(frames(directory)) == 20
        if arm in ("learned", "skip", "always"):
            trajectories.append(frames(directory))
            assert result.decisions == 10 and result.eligible == result.accepted == 0
            for i in range(2):
                loaded = TrustDirectionLearner.load(directory / f"tick-0020-ant-{i:02d}.npz", DirectionMotor(41), 0)
                assert loaded.writes == 0 and not loaded.weights().any()
        if arm == "rules":
            assert not list(directory.glob("*.npz"))
    assert trajectories[0] == trajectories[1] == trajectories[2]
    with pytest.raises(FileExistsError):
        run_world(plan, 18999, plan.conditions[0], "skip", tmp_path / "skip")


def test_worlds_use_same_source_schedule_without_shared_parameters(plan, tmp_path):
    condition = Condition(name="periodic", disturbance=DisturbanceConfig(response=(0,0,0,1,0,0,0,0),
        signal_radius=8., injury_per_step=0., speed_multiplier=.3, slowdown_radius=2.,
        period_steps=8, active_steps=4, inactive_signal_scale=0., motion_amplitude=.5, motion_period_steps=8))
    schedules = []
    for arm in ("learned", "skip", "always", "mlp", "rules"):
        directory = tmp_path / arm
        result = run_world(plan, 18999, condition, arm, directory)
        schedules.append([(f.source_active, f.source_position) for f in frames(directory)])
        if arm == "skip":
            assert result.writes == [0, 0]
        if arm == "always":
            assert result.eligible > 0 and sum(result.writes) > 0
        if arm in ("learned", "skip", "always"):
            updates = [UpdateRecord.model_validate_json(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            assert len(updates) == result.decisions
            assert sum(r.changed for r in updates) == sum(result.writes)
            assert all(not r.accepted for r in updates if r.terminal)
    assert all(schedule == schedules[0] for schedule in schedules)
    actors = make_actors(plan, 18999, "learned")
    assert all(isinstance(a, OnlineForager) for a in actors)
    assert {p.data_ptr() for p in actors[0].frozen_parameters()}.isdisjoint(p.data_ptr() for p in actors[1].frozen_parameters())
    assert actors[0].agent.offset.data_ptr() != actors[1].agent.offset.data_ptr()


def test_rule_world_never_constructs_or_calls_network(plan, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("规则世界使用神经网络")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    result = run_world(plan, 18999, plan.conditions[0], "rules", tmp_path / "rules")
    assert result.decisions == result.accepted == 0


def test_death_finishes_pending_windows_without_writing(plan, tmp_path):
    condition = Condition(name="moving-danger", disturbance=DisturbanceConfig(contact_radius=20., injury_per_step=1., signal_radius=20.))
    result = run_world(plan, 18999, condition, "always", tmp_path / "danger")
    assert result.steps == 1 and result.deaths == 2 and result.exhausted == 2
    assert result.decisions == 2 and result.accepted == 0 and result.writes == [0, 0]


@pytest.mark.parametrize("changes", [{"seeds": ()}, {"seeds": (1,1)}, {"conditions": ()},
    {"adaptation": TrustConfig(feedback_mode="critic")}])
def test_invalid_continuous_plan_rejected(plan, changes):
    with pytest.raises(ValidationError):
        ContinuousPlan.model_validate({**plan.model_dump(), **changes})


def test_reconstruction_compares_full_trace_proposals_and_parameters(plan, tmp_path):
    import numpy as np
    from mathhackson.training.comparison.continuous_audit import compare_world, window_scores
    original, rebuilt = tmp_path / "original", tmp_path / "rebuilt"
    first = run_world(plan, 18999, plan.conditions[0], "learned", original)
    second = run_world(plan, 18999, plan.conditions[0], "learned", rebuilt)
    parameters, proposals = compare_world(original, rebuilt, first, second)
    assert parameters == 18 and proposals == 10
    windows = window_scores(first, frames(original))
    assert windows[0].active_individual_steps == 40
    assert windows[1].active_individual_steps == windows[2].active_individual_steps == 0
    path = rebuilt / "updates.jsonl"
    contents = path.read_text()
    path.write_text(contents.replace('"accepted":false', '"accepted":true', 1))
    with pytest.raises(ValueError, match="提案"):
        compare_world(original, rebuilt, first, second)
    path.write_text(contents)
    snapshot = rebuilt / "tick-0020-ant-00.residual.npz"
    with np.load(snapshot, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values["fast"][0] += .01
    np.savez(snapshot, **values)
    with pytest.raises(ValueError, match="参数快照"):
        compare_world(original, rebuilt, first, second)


def test_fixed_windows_keep_terminal_counts_without_fabricated_actions(plan, tmp_path):
    from mathhackson.training.comparison.continuous_audit import window_scores
    condition = Condition(name="moving-danger", disturbance=DisturbanceConfig(contact_radius=20., injury_per_step=1.))
    original = tmp_path / "dead"
    result = run_world(plan, 18999, condition, "always", original)
    trace = frames(original)
    windows = window_scores(result, trace)
    assert windows[0].deaths == windows[0].exhausted == 2
    assert all(w.deaths == w.exhausted == w.active_individual_steps == w.writes == 0 for w in windows[1:])
    trace[0].ants[0].reward += 1.
    with pytest.raises(ValueError, match="汇总"):
        window_scores(result, trace)


def test_acceptance_reads_exact_parameters_without_constructing_networks(plan, tmp_path, monkeypatch):
    from fastapi import HTTPException
    from mathhackson.training.comparison.continuous import run
    from mathhackson.training.comparison import acceptance
    directory = tmp_path / "run"
    run(plan, directory, protocol_sha256="test")
    tape = acceptance.TapeStore(directory)
    monkeypatch.setattr(acceptance, "store", lambda: tape)
    def forbidden(*args, **kwargs):
        raise AssertionError("只读验收创建或调用神经模型")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    header = acceptance.header("reference", 18999, "learned")
    assert len(header.initial_positions) == 2
    response = acceptance.trace("reference", 18999, "learned")
    with gzip.open(response.path, "rt") as stream:
        assert len(stream.readlines()) == 20
    weights = acceptance.weights("reference", 18999, "learned", 0)
    assert sum(len(g.points[0].values) for g in weights) == 1697
    assert weights[0].name == "direction.offset" and not weights[0].frozen
    assert all(g.frozen for g in weights[1:])
    assert all(g.frozen for g in acceptance.weights("reference", 18999, "skip", 0))
    assert acceptance.weights("reference", 18999, "rules", 0) == []
    with pytest.raises(HTTPException) as invalid_ant:
        acceptance.weights("reference", 18999, "rules", 2)
    assert invalid_ant.value.status_code == 404
    with pytest.raises(HTTPException) as invalid_world:
        acceptance.trace("reference", 2, "learned")
    assert invalid_world.value.status_code == 404
