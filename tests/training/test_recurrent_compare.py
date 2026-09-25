from pathlib import Path

import numpy as np
import pytest

from scripts.training.recurrent_compare import compare, read_parameters
from mathhackson.training.recurrent_session import RecurrentSession


def test_comparison_uses_paired_seeds_and_keeps_signed_differences(tmp_path: Path):
    session = RecurrentSession(tmp_path)
    while not session.history:
        session.step()
    checkpoint = session.directory / session.history[-1].checkpoint
    result = compare(checkpoint, first_seed=41, seeds=3)
    trials = {(trial["seed"], trial["task"], trial["mode"]): trial
              for trial in result["trials"]}
    assert len(trials) == 27
    assert len(result["pairs"]) == 18
    for seed in range(41, 44):
        perturbations = {trials[seed, "sensor", mode]["perturbation"]
                         for mode in ("off", "learned", "always")}
        assert len(perturbations) == 1
        assert .55 <= abs(perturbations.pop()) <= .75
    for pair in result["pairs"]:
        current = trials[pair["seed"], pair["task"], pair["mode"]]
        baseline = trials[pair["seed"], pair["task"], "off"]
        assert pair["step_difference"] == current["steps"] - baseline["steps"]
        assert pair["reward_difference"] == pytest.approx(current["reward"] - baseline["reward"])
    assert all(record["total"] == 3 for record in result["summary"])
    assert all(record["mean_writes"] == 0. for record in result["summary"] if record["mode"] == "off")


def test_comparison_rejects_legacy_checkpoint(tmp_path: Path):
    path = tmp_path / "not-recurrent.npz"
    np.savez(path, weights=np.zeros((39, 16)))
    with pytest.raises(ValueError, match="缺少循环模型参数"):
        read_parameters(path)


def test_comparison_rejects_previous_recurrent_shape(tmp_path: Path):
    path = tmp_path / "old-recurrent.npz"
    np.savez(path, motor=np.zeros((2, 16)), input_weights=np.zeros((4, 16)),
             hidden_weights=np.zeros((4, 4)), hidden_bias=np.zeros(4),
             action_weights=np.zeros((2, 4)), gate_weights=np.zeros(5),
             write_weights=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="旧版循环模型"):
        read_parameters(path)
