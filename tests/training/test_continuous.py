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
    for i in range(8):
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


@pytest.mark.parametrize("rate", [0., .1])
def test_reward_baseline_history_matches_real_snapshots_and_detects_corruption(plan, tmp_path, rate):
    import numpy as np
    from mathhackson.training.comparison.baseline_audit import history
    plan = plan.model_copy(update={"respawn": True, "adaptation": plan.adaptation.model_copy(
        update={"historical_baseline_rate": rate})})
    directory = tmp_path / "baseline"
    world = run_world(plan, 18999, plan.conditions[0], "skip", directory)
    result = history(directory, world, plan)
    assert result.windows == (10 if rate else 0)
    assert result.snapshots == 8
    path = directory / "tick-0008-ant-00.residual.npz"
    with np.load(path, allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    fields['return_baseline'] = np.asarray(999.)
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="快照"):
        history(directory, world, plan)


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


def test_acceptance_separates_preview_from_registered_results(plan, tmp_path, monkeypatch):
    import json
    from fastapi import HTTPException
    from mathhackson.training.comparison.continuous import run
    from mathhackson.training.comparison import acceptance
    directory = tmp_path / "distant"
    run(plan, directory, protocol_sha256="engineering-test")
    tape = acceptance.TapeStore(directory)
    monkeypatch.setattr(acceptance, "store", lambda: tape)
    monkeypatch.setattr(acceptance, "DISTANT_DATA", directory)
    monkeypatch.setattr(acceptance, "REVIVING_DATA", tmp_path / "missing-revival")
    acceptance.reviving_store.cache_clear()
    acceptance.distant_store.cache_clear()
    try:
        assert [batch.id for batch in acceptance.batches()] == ["distant", "registered"]
        preview = json.loads(acceptance.catalog("distant").body)
        assert preview["summary"] is None
        assert preview["execution"]["protocol_sha256"] == "engineering-test"
        assert acceptance.catalog().path == acceptance.RESULT
        assert acceptance.header("reference", 18999, "rules", "distant").stock == 48
        assert acceptance.weights("reference", 18999, "rules", 0, "distant") == []
        with pytest.raises(HTTPException):
            acceptance.selected_store("../../other")
        lines = (directory / "worlds.jsonl").read_text().splitlines()
        (directory / "worlds.jsonl").write_text("\n".join(lines[:-1]))
        acceptance.distant_store.cache_clear()
        assert [batch.id for batch in acceptance.batches()] == ["registered"]
        with pytest.raises(HTTPException) as incomplete:
            acceptance.catalog("distant")
        assert incomplete.value.status_code == 409
    finally:
        acceptance.distant_store.cache_clear()


def test_revival_run_preserves_memory_and_records_repeated_deaths(plan, tmp_path):
    import numpy as np
    condition = Condition(name="moving-danger", disturbance=DisturbanceConfig(
        contact_radius=20., injury_per_step=1., signal_radius=20.))
    plan = plan.model_copy(update={"respawn": True, "environment": plan.environment.model_copy(update={"horizon": 6})})
    result = run_world(plan, 18999, condition, "always", tmp_path / "revival")
    assert result.steps == 6 and result.deaths == result.exhausted == 12 and result.revivals == 10
    trace = frames(tmp_path / "revival")
    assert trace[-1].ants[0].cumulative_deaths == 6 and trace[-1].ants[0].revivals == 5
    assert trace[-1].ants[0].pending and trace[-1].food_stock == 48
    with np.load(tmp_path / "revival/tick-0006-ant-00.memory.npz") as state:
        assert state["history"].shape == (6, 8)
    records = [UpdateRecord.model_validate_json(line) for line in
               (tmp_path / "revival/updates.jsonl").read_text().splitlines()]
    assert any(r.changed and r.continuing_after_death for r in records)
    assert all(not r.accepted for r in records if r.tick == 6)


def test_revival_rule_world_never_uses_a_neural_network(plan, tmp_path, monkeypatch):
    plan = plan.model_copy(update={"respawn": True, "environment": plan.environment.model_copy(update={"horizon": 3})})
    condition = Condition(name="moving-danger", disturbance=DisturbanceConfig(contact_radius=20., injury_per_step=1.))
    def forbidden(*args, **kwargs):
        raise AssertionError("复活规则世界使用了神经网络")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    result = run_world(plan, 18999, condition, "rules", tmp_path / "rules-revival")
    assert result.deaths == 6 and result.revivals == 4
    assert result.decisions == 0 and not list((tmp_path / "rules-revival").glob("*.npz"))


def test_expanded_population_has_independent_parameters_and_random_streams(plan):
    expanded = plan.model_copy(update={"environment": plan.environment.model_copy(update={"ants": 32}), "respawn": True})
    actors = make_actors(expanded, 18999, "learned")
    assert len(actors) == 32
    for first, second in ((actors[0], actors[8]), (actors[8], actors[16]), (actors[16], actors[24])):
        assert all(a.equal(b) for a, b in zip(first.frozen_parameters(), second.frozen_parameters(), strict=True))
        assert {p.data_ptr() for p in first.frozen_parameters()}.isdisjoint(p.data_ptr() for p in second.frozen_parameters())
        assert first.agent.offset.data_ptr() != second.agent.offset.data_ptr()
        assert not first.agent.random.get_state().equal(second.agent.random.get_state())


def test_process_workers_reproduce_serial_worlds_and_order(plan, tmp_path):
    from mathhackson.training.comparison.continuous import Execution, WorldResult, run
    from mathhackson.training.comparison.continuous_audit import compare_world
    plan = plan.model_copy(update={"environment": plan.environment.model_copy(update={"horizon": 4})})
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    run(plan, serial, protocol_sha256="workers-test")
    run(plan, parallel, protocol_sha256="workers-test", workers=2)
    assert Execution.model_validate_json((parallel / "execution.json").read_text()).workers == 2
    first = [WorldResult.model_validate_json(line) for line in (serial / "worlds.jsonl").read_text().splitlines()]
    second = [WorldResult.model_validate_json(line) for line in (parallel / "worlds.jsonl").read_text().splitlines()]
    assert [row.arm for row in first] == [row.arm for row in second] == list(ARMS)
    for a, b in zip(first, second, strict=True):
        key = f"{a.condition}-{a.seed}-{a.arm}"
        compare_world(serial / key, parallel / key, a, b)
