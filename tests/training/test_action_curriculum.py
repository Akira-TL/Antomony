import gzip
import json

import numpy as np
import pytest
import torch

from test_basic_update_learning import plan as old_plan, record
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging import candidate_probe
from mathhackson.training.foraging.action_features import ActionFeatureRecord, action_effect_features
from mathhackson.training.foraging.candidate_probe import CandidateRecord, ProbePlan
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.update_curriculum import CurriculumPlan, fit_individual, read_partition, world_key
from mathhackson.training.foraging.update_decision import UpdateDecision


def action_plan():
    values = old_plan().model_dump()
    values.update(train_seeds=(11101,), held_seeds=(11111,), initial_norms=(0.,),
                  training_steps=4, checkpoint_every=2, decision_profile="actions")
    values["probe"].update(seeds=(11101, 11111), include_action_features=True)
    values["probe"]["adaptation"]["window"] = 4
    return CurriculumPlan.model_validate(values)


def test_plan_requires_explicit_recording_and_old_defaults_are_unchanged():
    old = old_plan()
    assert old.decision_profile == "parameters" and not old.probe.include_action_features
    with pytest.raises(ValueError, match="显式采集"):
        CurriculumPlan.model_validate(old.model_dump() | {"decision_profile": "actions"})
    values = action_plan().probe.model_dump()
    values.update(candidate_kind="gradient", adaptation={"window": 4})
    with pytest.raises(ValueError, match="只支持"):
        ProbePlan.model_validate(values)


@pytest.mark.parametrize("change", [
    {"values": [0.] * 37}, {"values": [0.] * 37 + [float("nan")]},
    {"values": [0.] * 37 + [1.]}, {"version": "unknown"},
])
def test_action_record_rejects_wrong_width_nonfinite_unbounded_or_unknown(change):
    with pytest.raises(ValueError):
        ActionFeatureRecord.model_validate({"values": [0.] * 38} | change)


def test_collection_records_before_branch_and_preserves_all_legacy_results(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(candidate_probe, "load_motor", lambda path: (DirectionMotor(41), None))
    monkeypatch.setattr(ForagingPolicy, "load", lambda path: ForagingPolicy(71).with_budgets())
    values = action_plan().probe.model_dump()
    values.update(include_zero_candidates=True, pairs_per_world=2, branch_horizon=2)
    values["environment"].update(ants=2, horizon=10)
    values["disturbance"].update(center_fraction=.1, signal_radius=3.)
    captured = []
    evaluated = []
    evaluate = candidate_probe.evaluate_candidate

    def capture(agent, observation):
        features = action_effect_features(agent, observation)
        captured.append(features.copy())
        return features

    def branch(*args, **kwargs):
        assert len(captured) == len(evaluated) + 1
        evaluated.append(True)
        return evaluate(*args, **kwargs)

    parents, samples, results = [], [], []
    for enabled in (False, True):
        config = ProbePlan.model_validate(values | {"include_action_features": enabled})
        directory = tmp_path / str(enabled)
        directory.mkdir()
        if enabled:
            monkeypatch.setattr(candidate_probe, "action_effect_features", capture)
            monkeypatch.setattr(candidate_probe, "evaluate_candidate", branch)
        results.append(candidate_probe.collect(config, 11101, "benign", directory))
        with gzip.open(directory / "parent.jsonl.gz", "rt") as stream:
            parents.append(stream.read())
        samples.append([json.loads(line) for line in (directory / "pairs.jsonl").read_text().splitlines()])
    assert parents[0] == parents[1] and results[0] == results[1]
    assert len(captured) == len(evaluated) == len(samples[1]) == 2
    assert any(row["result"]["changed"] for row in samples[1])
    assert any(np.any(values[16:32]) for values in captured)
    assert all("action_effects" not in row for row in samples[0])
    for old, new, preview in zip(*samples, captured, strict=True):
        parsed = CandidateRecord.model_validate(new)
        np.testing.assert_array_equal(parsed.action_effects.values, preview)
        del new["action_effects"]
        assert new == old


def test_reader_and_individual_fit_use_action_features_and_save_new_format(tmp_path):
    plan = action_plan()
    directory = tmp_path / world_key(0, 11101)
    directory.mkdir()
    path = directory / "pairs.jsonl"
    base = record()
    path.write_text(base.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="不得补零"):
        read_partition(tmp_path, plan, "train")
    examples = []
    for focal, sign in ((0, 1.), (1, -1.)):
        item = base.model_copy(update={"focal": focal, "action_effects": ActionFeatureRecord(values=[.1] * 38),
            "result": base.result.model_copy(update={"accept": base.result.accept.model_copy(update={"focal_reward": sign})})})
        examples.append(item)
    path.write_text("\n".join(item.model_dump_json() for item in examples) + "\n")
    # 留出目录不存在，不能为了训练读取它。
    rows = read_partition(tmp_path, plan, "train")
    assert len(rows) == 2 and all(row.features.shape == (178,) for row in rows)
    for focal, sign in ((0, 1.), (1, -1.)):
        target = tmp_path / f"ant-{focal}"
        fitted = fit_individual(rows, focal, plan, target)
        assert fitted.examples == 1 and fitted.steps == 4
        assert sorted(p.name for p in target.glob("*.npz")) == ["step-0000.npz", "step-0002.npz", "step-0004.npz"]
        model = UpdateDecision.load(target / "step-0004.npz")
        assert model.profile == "actions" and model(torch.from_numpy(rows[focal].features)).item() * sign > 0.
    legacy = CurriculumPlan.model_validate(plan.model_dump() | {"decision_profile": "parameters"})
    old_rows = read_partition(tmp_path, legacy, "train")
    for current, old in zip(rows, old_rows, strict=True):
        np.testing.assert_array_equal(current.features[:140], old.features)
