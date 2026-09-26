from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.foraging.adaptation import AdaptationProposal
from mathhackson.training.foraging.candidate_probe import CandidateRecord
from mathhackson.training.foraging.candidate_value import BranchOutcome, CandidateValue
from mathhackson.training.foraging.trust_candidate import CandidateDiagnostics
from mathhackson.training.foraging.update_curriculum import CurriculumPlan, LabeledDecision, fit_individual, read_partition, world_key
from mathhackson.training.foraging.update_decision import INPUT_WIDTH, UpdateDecision


def plan():
    return CurriculumPlan.model_validate_json(Path(".research/protocols/basic-update-learning.json").read_text())


def record(seed=11101, focal=0):
    outcome = BranchOutcome(steps=3, focal_reward=0., colony_reward=0., focal_deliveries=0,
                            colony_deliveries=0, focal_death=False, colony_deaths=0,
                            focal_exhausted=False, focal_injury=0.)
    return CandidateRecord(condition="benign", seed=seed, tick=16, focal=focal, observation=[0.] * 78,
                           hidden=[0.] * 8, proposal=AdaptationProposal(1, (.01,) * 45, .1, 0., 0., .5, 16),
                           diagnostics=CandidateDiagnostics(.01, .02, .01, 0),
                           result=CandidateValue(skip=outcome, accept=outcome.model_copy(update={"focal_reward": 1.}),
                                                 changed=True, accepted_weights=[.01] * 45))


def test_frozen_plan_rejects_overlap_danger_and_unsafe_perturbation():
    original = plan()
    for change in ({"held_seeds": (11101,)}, {"initial_norms": (4.,)}, {"train_seeds": ()}):
        with pytest.raises(ValueError):
            CurriculumPlan.model_validate(original.model_dump() | change)
    probe = original.probe.model_dump()
    probe["disturbance"]["injury_per_step"] = .1
    with pytest.raises(ValueError):
        CurriculumPlan.model_validate(original.model_dump() | {"probe": probe})


def test_reader_opens_only_requested_worlds_and_rejects_wrong_identity(tmp_path):
    config = plan()
    for index, _ in enumerate(config.initial_norms):
        for seed in config.train_seeds:
            path = tmp_path / world_key(index, seed)
            path.mkdir()
            (path / "pairs.jsonl").write_text(record(seed).model_dump_json() + "\n")
    # 留出目录不存在，训练读取仍必须成功。
    rows = read_partition(tmp_path, config, "train")
    assert len(rows) == 12 and all(row.benefit == 1. for row in rows)
    path = tmp_path / world_key(0, config.train_seeds[0]) / "pairs.jsonl"
    path.write_text(record(11111).model_dump_json() + "\n")
    with pytest.raises(ValueError):
        read_partition(tmp_path, config, "train")


def test_fit_keeps_individual_data_separate_saves_schedule_and_rejects_held(tmp_path):
    config = plan().model_copy(update={"training_steps": 4, "checkpoint_every": 2})
    x = np.zeros(INPUT_WIDTH, dtype=np.float32)
    x[0] = 1.
    rows = [LabeledDecision(world_key(0, 11101), i, x, sign, record(focal=i)) for i, sign in ((0, 1.), (1, -1.))]
    a = fit_individual(rows, 0, config, tmp_path / "a")
    b = fit_individual(rows, 1, config, tmp_path / "b")
    assert a.positive == b.negative == 1 and a.examples == b.examples == 1
    assert len(list((tmp_path / "a").glob("*.npz"))) == 3
    assert UpdateDecision.load(tmp_path / "a/step-0004.npz")(torch.from_numpy(x)).item() > 0.
    assert UpdateDecision.load(tmp_path / "b/step-0004.npz")(torch.from_numpy(x)).item() < 0.
    invalid = [LabeledDecision(world_key(0, 11111), 0, x, 1., record(11111))]
    with pytest.raises(ValueError):
        fit_individual(invalid, 0, config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_empty_individual_is_explicitly_untrained(tmp_path):
    config = plan()
    result = fit_individual([], 0, config, tmp_path / "empty")
    assert result.steps == 0 and result.final_loss is None and result.examples == 0
    assert UpdateDecision.load(tmp_path / "empty/step-0200.npz").updates == 0
