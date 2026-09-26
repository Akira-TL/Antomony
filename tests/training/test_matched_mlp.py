import numpy as np
import pytest
import torch
from pydantic import ValidationError

from mathhackson.training.comparison.run import Config, train_signal
from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy, NOVEL_COLUMNS


@pytest.mark.parametrize("width,count", [(14, 1365), (17, 1650)])
def test_width_roundtrip_and_history_independence(tmp_path, width, count):
    model = FeedforwardPolicy(81, hidden_width=width)
    assert sum(p.numel() for p in model.parameters()) == count
    obs, _ = signal_batch(np.random.default_rng(91), 32, budgets=True)
    expected = model(obs)
    path = tmp_path / "model.npz"
    model.save(path, update=2, phase="signal")
    loaded = FeedforwardPolicy.load(path)
    assert loaded.hidden_width == width
    assert not any(p.requires_grad for p in loaded.parameters())
    for actual, wanted in zip(loaded(obs, (torch.randn(32, 8),) * 16), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0., atol=0.)
    with np.load(path) as saved:
        assert str(saved["version"]) == ("local-mlp-v1" if width == 14 else "local-mlp-v2")
    with pytest.raises(FileExistsError):
        model.save(path, update=3, phase="signal")


def test_default_initialization_matches_original_architecture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        encoder = torch.nn.Linear(86, 14)
        middle = torch.nn.Linear(14, 8)
        direction = torch.nn.Linear(8, 2)
        value = torch.nn.Linear(8, 1)
        with torch.no_grad():
            encoder.weight[:, NOVEL_COLUMNS] = 0.
            direction.weight.mul_(.1)
            direction.bias.copy_(torch.tensor([1., 0.]))
    model = FeedforwardPolicy(81)
    for old, actual in zip((encoder, middle, direction, value),
                           (model.encoder, model.middle, model.direction, model.value), strict=True):
        for before, after in zip(old.parameters(), actual.parameters(), strict=True):
            torch.testing.assert_close(before, after, rtol=0., atol=0.)


def test_added_units_receive_gradients_without_changing_other_individual():
    model = FeedforwardPolicy(81, hidden_width=17)
    other = FeedforwardPolicy(82, hidden_width=17)
    before = [p.detach().clone() for p in other.parameters()]
    obs, target = signal_batch(np.random.default_rng(91), 256, budgets=True)
    loss = 1. - (model(obs)[0] * target).sum(-1).mean()
    loss.backward()
    assert bool((model.encoder.weight.grad.abs().sum(dim=1) > 0.).all())
    assert bool((model.middle.weight.grad.abs().sum(dim=0) > 0.).all())
    assert model.value.weight.grad is None and model.value.bias.grad is None
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003).step()
    model.assert_reserved()
    assert all(a.equal(b) for a, b in zip(before, other.parameters(), strict=True))


@pytest.mark.parametrize("width", [0, -1, 17.5, True])
def test_invalid_width_is_rejected(width):
    with pytest.raises(ValueError):
        FeedforwardPolicy(1, hidden_width=width)
    with pytest.raises(ValidationError):
        Config(source_commit="test", hidden_width=width)


@pytest.mark.parametrize("width", [np.array(14), np.array(17.5), np.array([17])])
def test_checkpoint_width_must_match_weights(tmp_path, width):
    path = tmp_path / "model.npz"
    FeedforwardPolicy(81, hidden_width=17).save(path, update=0, phase="signal")
    with np.load(path) as saved:
        content = {key: saved[key].copy() for key in saved.files}
    content["hidden_width"] = width
    bad = tmp_path / "bad.npz"
    np.savez(bad, **content)
    with pytest.raises(ValueError):
        FeedforwardPolicy.load(bad)


def test_memory_cannot_silently_save_incompatible_width():
    with pytest.raises(ValueError, match="14"):
        MemoryPolicy(FeedforwardPolicy(81, hidden_width=17))


def test_training_entry_uses_width_and_retains_snapshots(tmp_path):
    config = Config(source_commit="test", hidden_width=17, model_seeds=(81, 82),
                    signal_updates=2, checkpoint_every=1, signal_batch_size=16)
    models = train_signal(config, tmp_path)
    assert len(models) == 2 and all(model.hidden_width == 17 for model in models)
    assert len((tmp_path / "signal.jsonl").read_text().splitlines()) == 4
    for seed, model in zip(config.model_seeds, models, strict=True):
        model.assert_reserved()
        for update in (0, 1, 2):
            path = tmp_path / f"seed-{seed}-signal-{update:06d}.npz"
            assert FeedforwardPolicy.load(path).hidden_width == 17
            if update:
                assert path.with_suffix(".optimizer.pt").is_file()
