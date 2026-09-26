from dataclasses import replace

import numpy as np
import pytest
import torch

from mathhackson.training.foraging.curriculum import signal_batch
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.rules import LocalRuleController
from mathhackson.training.foraging.signals import receptor_points


def test_mlp_parameter_budget_history_independence_and_serialization(tmp_path):
    model = FeedforwardPolicy(81)
    assert sum(p.numel() for p in model.parameters()) == 1365
    obs, _ = signal_batch(np.random.default_rng(91), 6, budgets=True)
    expected = model(obs)[0]
    actual = model(obs, tuple(torch.ones(6, 8) for _ in range(16)))[0]
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    model.save(tmp_path / "mlp.npz", update=0, phase="signal")
    loaded = FeedforwardPolicy.load(tmp_path / "mlp.npz")
    torch.testing.assert_close(loaded(obs)[0], expected, rtol=0., atol=0.)
    with pytest.raises(ValueError):
        model(torch.zeros(80))


def test_mlp_training_keeps_reserved_inputs_zero_and_other_individual_unchanged():
    model, other = FeedforwardPolicy(81), FeedforwardPolicy(81)
    obs, target = signal_batch(np.random.default_rng(91), 32, budgets=True)
    before = [p.detach().clone() for p in other.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    loss = 1. - (model(obs)[0] * target).sum(-1).mean()
    loss.backward()
    optimizer.step()
    model.assert_reserved()
    assert all(a.equal(b) for a, b in zip(before, other.parameters(), strict=True))
    assert any(not a.equal(b) for a, b in zip(model.parameters(), other.parameters(), strict=True))


def local_observation():
    points = receptor_points(np.zeros(2, dtype=np.float32), 0.)
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 0] = 1. + points[:, 1]
    receptors[:, 1] = 1. - points[:, 1]
    receptors[:, 2] = 1. + points[:, 0]
    return LocalObservation(receptors, False, False, True, 0., (1., 1.))


def test_rule_signal_selection_and_return_budget_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("规则组调用了神经网络")
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    obs = local_observation()
    assert LocalRuleController(1).act(obs).turn == 1.
    assert LocalRuleController(1).act(replace(obs, carrying=True)).turn == -1.
    assert LocalRuleController(1).act(replace(obs, budgets=(0., 1.))).turn == -1.
    receptors = obs.receptors.copy()
    receptors[:, 0] = 0.
    action = LocalRuleController(1).act(replace(obs, receptors=receptors))
    assert action.turn == 0. and action.move


def test_rule_ignores_untrained_responses_and_reproducible_exploration():
    obs = local_observation()
    receptors = obs.receptors.copy()
    receptors[:, 3:] = 1000.
    assert LocalRuleController(3).act(obs) == LocalRuleController(3).act(replace(obs, receptors=receptors))
    empty = replace(obs, receptors=np.zeros((9, 8), dtype=np.float32))
    first, second = LocalRuleController(31), LocalRuleController(31)
    one = [first.act(empty) for _ in range(40)]
    two = [second.act(empty) for _ in range(40)]
    assert one == two and all(a.move and -1. <= a.turn <= 1. for a in one)
    assert len({a.turn for a in one}) > 1
