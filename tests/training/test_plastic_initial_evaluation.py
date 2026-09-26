"""初始参照入口的机械测试；临时合成输出不作为研究结果。"""
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.feedback_cues import CueConfig
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.plastic_course import run
from mathhackson.training.foraging.plastic_direction import PlasticDirection


def synthetic_plan(**options):
    source = run.Source(path="unused-synthetic-model.npz", sha256="a" * 64)
    return run.Protocol(motor=source, updates=25, evaluation_batches=1,
        course=CueConfig(steps=8, batch=1), initializations=(
            run.Initialization(seed=11, foundation=source, train_seed=100, evaluation_seed=200),
            run.Initialization(seed=12, foundation=source, train_seed=300, evaluation_seed=400)), **options)


def test_old_protocol_does_not_enable_initial_evaluation():
    old = synthetic_plan().model_dump(exclude={"evaluate_initial"})
    assert run.Protocol(**old).evaluate_initial is False
    assert synthetic_plan(evaluate_initial=True).evaluate_initial is True


@pytest.mark.parametrize("structure_update", [None, 0])
def test_evaluate_writes_selected_structure_marker_only(tmp_path: Path, structure_update):
    plan = synthetic_plan()
    model = PlasticDirection(DirectionMotor(8), seed=9)
    base = MemoryPolicy(FeedforwardPolicy(10))
    run.evaluate(model, base, plan.initializations[0], plan, tmp_path,
        time.monotonic() + 30, structure_update=structure_update)
    archives = list(tmp_path.glob("*.npz"))
    assert len(archives) == 3 * 4
    for path in archives:
        with np.load(path, allow_pickle=False) as trace:
            assert int(trace["structure_update"]) == (plan.updates if structure_update is None else 0)
            assert trace["rewards"].shape == (8, 1)


@pytest.mark.parametrize("evaluate_initial", [False, True])
def test_main_keeps_initial_copy_separate_and_reuses_one_budget(tmp_path: Path, monkeypatch, evaluate_initial):
    plan = synthetic_plan(evaluate_initial=evaluate_initial, direction_mode="bounded")
    plan_path, output = tmp_path / "synthetic-plan.json", tmp_path / "output"
    stored = plan.model_dump_json().encode()
    plan_path.write_bytes(stored)
    monkeypatch.setattr(sys, "argv", ["synthetic-test", "--output", str(output),
        "--plan", str(plan_path), "--freeze-commit", "synthetic-freeze"])

    def fake_git(command, **options):
        if command[:2] == ["git", "show"]:
            return stored
        assert command == ["git", "rev-parse", "HEAD"] and options.get("text") is True
        return "synthetic-code\n"

    monkeypatch.setattr(run.subprocess, "check_output", fake_git)
    monkeypatch.setattr(run, "verified", lambda source: Path(source.path))
    monkeypatch.setattr(run, "load_motor", lambda path: (DirectionMotor(8), None))
    monkeypatch.setattr(run.MemoryPolicy, "load", lambda path: MemoryPolicy(FeedforwardPolicy(10)))
    calls: list[tuple[str, int, PlasticDirection, float]] = []

    def fake_evaluate(model, base, initialization, actual_plan, directory, deadline, *, structure_update=None):
        assert actual_plan is not None and model.direction_mode == "bounded"
        kind = "initial" if structure_update == 0 else "final"
        calls.append((kind, initialization.seed, model, deadline))
        if kind == "initial":
            assert directory == output / "untrained" / f"seed-{initialization.seed}"
            with np.load(directory / "step-0000.npz", allow_pickle=False) as archive:
                assert int(archive["update"]) == 0
                assert str(archive["direction_mode"]) == "bounded"
        else:
            assert structure_update is None
            assert directory == output / f"seed-{initialization.seed}"
        model.eval().requires_grad_(False)

    def fake_train(model, base, initialization, actual_plan, directory, deadline):
        calls.append(("train", initialization.seed, model, deadline))
        assert model.training and model.direction_mode == "bounded"
        assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 155
        if evaluate_initial:
            initial = next(item[2] for item in calls if item[:2] == ("initial", initialization.seed))
            assert initial is not model and not initial.training
            for before, after in zip(initial.parameters(), model.parameters()):
                assert before.data_ptr() != after.data_ptr()
                assert torch.equal(before, after)
                assert not before.requires_grad
            with torch.no_grad():
                model.encoder.bias.add_(1.)
            assert not torch.equal(initial.encoder.bias, model.encoder.bias)

    monkeypatch.setattr(run, "evaluate", fake_evaluate)
    monkeypatch.setattr(run, "train", fake_train)
    run.main()
    expected = ["initial", "train", "final"] if evaluate_initial else ["train", "final"]
    assert [item[0] for item in calls] == expected * 2
    assert len({item[3] for item in calls}) == 1
    assert (output / "untrained").exists() == evaluate_initial
    for initialization in plan.initializations:
        trained = next(item[2] for item in calls if item[:2] == ("train", initialization.seed))
        final = next(item[2] for item in calls if item[:2] == ("final", initialization.seed))
        assert trained is final
