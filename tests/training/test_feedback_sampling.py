import gzip
from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging import candidate_probe
from mathhackson.training.foraging.candidate_probe import CandidateRecord, ProbePlan
from mathhackson.training.foraging.policy import ForagingPolicy


def test_interval_requires_aligned_feedback_windows():
    plan = ProbePlan.model_validate_json(Path(".research/protocols/restoration-coherent.json").read_text())
    with pytest.raises(ValueError):
        ProbePlan.model_validate(plan.model_dump() | {"sample_interval": 17})
    aligned = ProbePlan.model_validate(plan.model_dump() | {"sample_interval": 64, "include_zero_candidates": True})
    assert aligned.sample_interval == 64


def test_matched_parent_paths_and_sample_identity_across_feedback_windows(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(candidate_probe, "load_motor", lambda path: (DirectionMotor(41), None))
    monkeypatch.setattr(ForagingPolicy, "load", lambda path: ForagingPolicy(71).with_budgets())
    original = ProbePlan.model_validate_json(Path(".research/protocols/restoration-coherent.json").read_text())
    base = original.model_dump()
    base.update(sample_interval=4, include_zero_candidates=True, branch_horizon=2, pairs_per_world=2)
    base["environment"].update(ants=2, horizon=10)
    # 没有陌生输入也要配对保留零候选，不能按各自非零集合换焦点。
    base["disturbance"]["signal_strength"] = 0.
    records = []
    parents = []
    for window in (2, 4):
        config = dict(base)
        config["adaptation"] = dict(base["adaptation"], window=window)
        plan = ProbePlan.model_validate(config)
        directory = tmp_path / str(window)
        directory.mkdir()
        result = candidate_probe.collect(plan, 71, "benign", directory)
        assert result.pairs == 2
        with gzip.open(directory / "parent.jsonl.gz", "rt") as stream:
            parents.append(stream.read())
        rows = [CandidateRecord.model_validate_json(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
        records.append(rows)
        assert all(not np.any(row.proposal.delta) and not row.result.changed for row in rows)
    assert parents[0] == parents[1]
    assert [(r.tick, r.focal) for r in records[0]] == [(4, 0), (8, 1)]
    for short, long in zip(*records, strict=True):
        assert (short.tick, short.focal, short.observation, short.hidden) == (long.tick, long.focal, long.observation, long.hidden)
        assert short.result.skip == long.result.skip
        assert short.proposal.steps == 2 and long.proposal.steps == 4
