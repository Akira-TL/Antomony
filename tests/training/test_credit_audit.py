import gzip
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mathhackson.training.comparison.credit_audit import (
    CandidateScore, Pair, audit_world, decide, describe, verify_pair,
)
from mathhackson.training.direction.checkpoint import MotorSnapshot, save_motor
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.candidate_probe import CandidateRecord, ProbePlan, collect
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy


def passing_pair(seed):
    old = CandidateScore(reward=describe([0., -1.]), focal_deliveries=describe([0., -1.]),
                         colony_deliveries=describe([0., -1.]), nonzero_candidates=1)
    new = CandidateScore(reward=describe([1., 0.]), focal_deliveries=describe([1., 0.]),
                         colony_deliveries=describe([1., 0.]), nonzero_candidates=1)
    return Pair(seed=seed, count=2, four=old, sixteen=new,
                reward_difference=describe([1., 1.]), delivery_difference=describe([1., 1.]))


def test_signed_descriptions_keep_empty_and_negative_values():
    assert describe([-3., 0., 1.]).mean == pytest.approx(-2 / 3)
    assert describe([-3., 0., 1.]).negative == 1
    assert describe([]).mean is None
    with pytest.raises(ValueError):
        describe([float("nan")])


@pytest.mark.parametrize("failure", [None, "no-positive", "fewer-positive", "reward-loss", "delivery-loss", "no-delivery", "tied", "empty"])
def test_continuation_needs_both_worlds_and_actual_delivery(failure):
    pairs = [passing_pair(1), passing_pair(2)]
    second = pairs[1]
    if failure == "no-positive":
        second.sixteen.reward = describe([0., 0.])
    elif failure == "fewer-positive":
        second.four.reward = describe([1., 1.])
    elif failure == "reward-loss":
        second.reward_difference = describe([-1., 0.])
    elif failure == "delivery-loss":
        second.delivery_difference = describe([-1., 0.])
    elif failure == "no-delivery":
        for pair in pairs:
            pair.sixteen.focal_deliveries = describe([0., 0.])
    elif failure == "tied":
        for pair in pairs:
            pair.reward_difference = describe([0., 0.])
    elif failure == "empty":
        second.count = 0
        second.reward_difference = describe([])
    assert decide(pairs) is (failure is None)


def test_duplicate_world_cannot_count_as_two_seeds():
    with pytest.raises(ValueError):
        decide([passing_pair(1), passing_pair(1)])


@pytest.fixture(params=[None, 16])
def sample(tmp_path, request):
    fields = ProbePlan.model_validate_json(Path(".research/protocols/credit-history-sixteen.json").read_text()).model_dump()
    fields.update(seeds=(1,), policy_directory=str(tmp_path), motor_path=str(tmp_path / "motor.npz"))
    fields["environment"].update(ants=1, horizon=20)
    fields["adaptation"]["credit_horizon"] = request.param
    plan = ProbePlan.model_validate(fields)
    MemoryPolicy(FeedforwardPolicy(1)).save(tmp_path / "episode-0000-ant-00.npz", update=0, phase="frozen")
    save_motor(tmp_path / "motor.npz", DirectionMotor(41), MotorSnapshot(seed=41, updates=0))
    directory = tmp_path / "sample"
    directory.mkdir()
    world = collect(plan, 1, "benign", directory)
    return directory, plan, world


@pytest.mark.parametrize("tamper", [None, "frames", "weights", "credit", "delta"])
def test_world_checks_frozen_weights_and_candidate_timing(sample, tamper):
    directory, plan, world = sample
    if tamper == "frames":
        path = directory / "parent.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            lines = stream.readlines()
        with gzip.open(path, "wt") as stream:
            stream.writelines(lines[:-1])
    elif tamper == "weights":
        path = directory / "tick-0020-ant-00.npz"
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        values["recent_weights"][0, 0] += .1
        np.savez(path, **values)
    elif tamper in ("credit", "delta"):
        path = directory / "pairs.jsonl"
        row = CandidateRecord.model_validate_json(path.read_text())
        if tamper == "credit":
            row.diagnostics = replace(row.diagnostics, credit_steps=4)
        else:
            row.proposal = replace(row.proposal, delta=tuple([1.] + list(row.proposal.delta[1:])))
        path.write_text(row.model_dump_json() + "\n")
    if tamper:
        with pytest.raises(ValueError):
            audit_world(directory, plan, world)
    else:
        rows, snapshots, _ = audit_world(directory, plan, world)
        assert snapshots == 2 and len(rows) == 1 and rows[0].tick == 16


@pytest.mark.parametrize("tamper", [None, "parent", "state", "before-action", "skip"])
def test_pairing_rejects_different_states_or_skip_branches(sample, tamper):
    directory, plan, world = sample
    rows, _, parent = audit_world(directory, plan, world)
    other = [r.model_copy(deep=True) for r in rows]
    if tamper == "state":
        other[0].observation[0] += .1
    elif tamper == "before-action":
        values = list(other[0].action_effects.values)
        values[32] += .1
        other[0].action_effects = other[0].action_effects.model_copy(update={"values": tuple(values)})
    elif tamper == "skip":
        outcome = other[0].result.skip
        other[0].result = other[0].result.model_copy(update={
            "skip": outcome.model_copy(update={"focal_reward": outcome.focal_reward + 1.})})
    if tamper:
        with pytest.raises(ValueError):
            verify_pair(rows, other, parent, "wrong" if tamper == "parent" else parent)
    else:
        verify_pair(rows, other, parent, parent)
