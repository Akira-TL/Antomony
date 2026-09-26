"""结构参数NPZ往返检查，不表示恢复完整在线过程。"""
from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.plastic_course.checkpoint import load_checkpoint, save_checkpoint
from mathhackson.training.foraging.plastic_direction import PlasticDirection


def test_structure_roundtrip_restores_values_and_forward(tmp_path: Path) -> None:
    original = PlasticDirection(DirectionMotor(41), seed=7, feature_width=6, max_step=.03, max_fast=.7)
    with torch.no_grad():
        original.recent_weights.fill_(.12)
        original.sparse_weights.fill_(-.13)
        original.decay_logit.fill_(.27)
    path = tmp_path / "step-0025.npz"
    save_checkpoint(path, original, update=25, seed=7)
    restored = load_checkpoint(path)
    assert restored.feature_width == 6 and restored.max_step == .03 and restored.max_fast == .7
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
        assert value.data_ptr() != restored.state_dict()[key].data_ptr()
    base, features = torch.tensor([[.6, .8]]), torch.ones(1, 6)
    left, right = original.initial_state(), restored.initial_state()
    for _ in range(5):
        a, b = original.infer(base, features, left), restored.infer(base, features, right)
        assert torch.equal(a.probabilities, b.probabilities) and a.actions == b.actions
        x = original.feedback(torch.ones(1), a.state)
        y = restored.feedback(torch.ones(1), b.state)
        assert torch.equal(x.modulation, y.modulation)
        assert torch.equal(x.state.fast, y.state.fast)
        left, right = x.state, y.state
    with np.load(path, allow_pickle=False) as data:
        assert str(data["version"]) == "plastic-direction-structure-v1"
        assert int(data["update"]) == 25 and int(data["seed"]) == 7
        assert "motor.encoder.weight" in data.files
        assert "fast" not in data.files and "history" not in data.files


def test_trainable_load_never_unfreezes_motor(tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    save_checkpoint(path, PlasticDirection(DirectionMotor(41)), update=0, seed=41)
    loaded = load_checkpoint(path, trainable=True)
    assert loaded.training and not loaded.motor.training
    assert all(not p.requires_grad if name.startswith("motor.") else p.requires_grad
               for name, p in loaded.named_parameters())
    assert not bool(loaded.initial_state().fast.any())
    assert loaded.initial_state().history == ()


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nonfinite", "version", "seed_dtype", "limit", "metadata_shape"])
def test_invalid_structure_archives_are_rejected(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "model.npz"
    save_checkpoint(path, PlasticDirection(DirectionMotor(41)), update=25, seed=41)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if damage == "missing":
        del arrays["encoder.bias"]
    elif damage == "extra":
        arrays["fast"] = np.zeros((1, 45, 2), dtype=np.float32)
    elif damage == "shape":
        arrays["encoder.bias"] = np.zeros(9, dtype=np.float32)
    elif damage == "dtype":
        arrays["encoder.bias"] = arrays["encoder.bias"].astype(np.float64)
    elif damage == "nonfinite":
        arrays["encoder.bias"][0] = np.nan
    elif damage == "version":
        arrays["version"] = np.asarray("unknown")
    elif damage == "seed_dtype":
        arrays["seed"] = np.int32(41)
    elif damage == "limit":
        arrays["max_step"] = np.float64(4.)
    else:
        arrays["update"] = np.asarray([25], dtype=np.int64)
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **arrays)
    with pytest.raises(ValueError):
        load_checkpoint(invalid)


def test_duplicate_archive_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    save_checkpoint(path, PlasticDirection(DirectionMotor(41)), update=0, seed=41)
    with zipfile.ZipFile(path, "a") as archive:
        content = archive.read("seed.npy")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("seed.npy", content)
    with pytest.raises(ValueError, match="重复"):
        load_checkpoint(path)


def test_save_refuses_overwrite_and_invalid_parameters(tmp_path: Path) -> None:
    path = tmp_path / "nested/model.npz"
    model = PlasticDirection(DirectionMotor(41))
    save_checkpoint(path, model, update=0, seed=41)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, update=25, seed=41)
    assert path.read_bytes() == original
    with torch.no_grad():
        model.decay_logit.fill_(float("inf"))
    bad = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="结构参数无效"):
        save_checkpoint(bad, model, update=25, seed=41)
    assert not bad.exists()


@pytest.mark.parametrize("update,seed", [(True, 1), (-1, 1), (1, False), (1, -1)])
def test_save_metadata_is_strict(tmp_path: Path, update: int, seed: int) -> None:
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.npz", PlasticDirection(DirectionMotor(41)), update=update, seed=seed)
