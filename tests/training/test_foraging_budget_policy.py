import numpy as np
import pytest
import torch

from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.policy import ForagingPolicy


def test_budget_extension_preserves_previous_outputs_and_does_not_share_parameters(tmp_path):
    original = ForagingPolicy(71)
    upgraded = original.with_budgets()
    inputs, _ = signal_batch(np.random.default_rng(7), 32, budgets=True)
    for before, after in zip(original(inputs), upgraded(inputs), strict=True):
        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)
    assert original.state_scale.weight.data_ptr() != upgraded.state_scale.weight.data_ptr()
    assert not upgraded.state_scale.weight[:, 1:].any()
    assert not upgraded.observation_layer.weight[:, 76:].any()
    path = tmp_path / "budget.npz"
    upgraded.save(path, update=0, phase="signal")
    restored = ForagingPolicy.load(path)
    assert restored.budget_inputs
    torch.testing.assert_close(restored(inputs)[0], upgraded(inputs)[0], rtol=0., atol=0.)
    with pytest.raises(ValueError):
        restored(inputs[:, :76])


def test_budget_curriculum_has_empty_return_and_reserved_inputs_remain_blank():
    inputs, targets = signal_batch(np.random.default_rng(8), 2048, budgets=True)
    assert inputs.shape == (2048, 78) and targets.shape == (2048, 2)
    empty_return = (inputs[:, 72] == 0.) & (inputs[:, 76] == 0.)
    assert empty_return.sum() > 400
    assert not inputs[:, :72].reshape(-1, 9, 8)[:, :, 3:].any()
    assert torch.all((inputs[:, 77] > 0.) & (inputs[:, 77] <= 1.))


def test_budget_training_can_update_new_inputs_but_not_reserved_or_other_ants():
    first, second = ForagingPolicy(71).with_budgets(), ForagingPolicy(71).with_budgets()
    first.set_phase("signal")
    frozen = [p.detach().clone() for p in second.parameters()]
    inputs, targets = signal_batch(np.random.default_rng(8), 64, budgets=True)
    optimizer = torch.optim.AdamW([p for p in first.parameters() if p.requires_grad], lr=.003)
    loss = 1. - (first(inputs)[0] * targets).sum(-1).mean()
    loss.backward()
    optimizer.step()
    assert first.observation_layer.weight[:, 76:].any()
    assert all(p.grad is None and not p.any() for p in first.reserved_parameters())
    assert all(a.equal(b) for a, b in zip(frozen, second.parameters(), strict=True))
