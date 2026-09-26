import math

import numpy as np
import pytest
import torch

from mathhackson.training.comparison.actors import NeuralForager
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.policy import RECENT_LAGS, SPARSE_LAGS


@pytest.mark.parametrize("batched", [False, True])
def test_zero_memory_is_bitwise_equal_for_arbitrary_history(batched):
    base = FeedforwardPolicy(31)
    model = MemoryPolicy(base)
    generator = torch.Generator().manual_seed(32)
    observation = torch.rand((7, 78) if batched else (78,), generator=generator)
    hidden_shape = (7, 8) if batched else (8,)
    history = tuple(torch.randn(hidden_shape, generator=generator) for _ in range(16))
    for length in (0, 1, 4, 8, 12, 16):
        assert all(torch.equal(a, b) for a, b in zip(base(observation), model(observation, history[:length]), strict=True))
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 64


@pytest.mark.parametrize("family,index,lag", [*(('recent_weights', i, lag) for i, lag in enumerate(RECENT_LAGS)),
                                            *(('sparse_weights', i, lag) for i, lag in enumerate(SPARSE_LAGS))])
def test_every_requested_lag_has_an_independent_effect(family, index, lag):
    model = MemoryPolicy(FeedforwardPolicy(0))
    with torch.no_grad():
        for parameter in model.base.parameters():
            parameter.zero_()
        getattr(model, family)[index, 0] = math.atanh(.5)
    history = tuple(torch.full((8,), float(i + 1)) for i in range(16))
    hidden = model(torch.zeros(78), history)[1]
    assert hidden[0] == pytest.approx(math.tanh(.5 * (17 - lag) / math.sqrt(8)), abs=1e-7)
    assert torch.count_nonzero(hidden[1:]) == 0


def test_memory_training_does_not_modify_cloned_base_or_other_individual():
    base = FeedforwardPolicy(1)
    first, second = MemoryPolicy(base), MemoryPolicy(base)
    original = [p.detach().clone() for p in base.parameters()]
    first.set_phase("reward")
    assert all(not p.requires_grad for p in first.base.parameters())
    assert all(p.requires_grad for p in first.memory_parameters())
    optimizer = torch.optim.AdamW(first.memory_parameters(), lr=.001)
    first(torch.ones(78), (torch.ones(8),))[1].sum().backward()
    optimizer.step()
    assert torch.count_nonzero(first.recent_weights)
    assert not torch.count_nonzero(second.recent_weights)
    assert all(a.equal(b) and a.equal(c) for a, b, c in zip(original, first.base.parameters(), base.parameters(), strict=True))
    assert {p.data_ptr() for p in first.parameters()}.isdisjoint(p.data_ptr() for p in second.parameters())


def test_checkpoint_roundtrip_does_not_reinterpret_old_mlp(tmp_path):
    base = FeedforwardPolicy(3)
    model = MemoryPolicy(base)
    with torch.no_grad():
        model.sparse_weights.fill_(.2)
    path = tmp_path / "memory.npz"
    model.save(path, update=2, phase="frozen")
    restored = MemoryPolicy.load(path)
    assert all(a.equal(b) for a, b in zip(model.parameters(), restored.parameters(), strict=True))
    old = tmp_path / "old.npz"
    base.save(old, update=1, phase="signal")
    with pytest.raises(ValueError):
        MemoryPolicy.load(old)
    with pytest.raises(ValueError):
        FeedforwardPolicy.load(path)
    with pytest.raises(FileExistsError):
        model.save(path, update=3, phase="frozen")


def test_invalid_history_is_rejected():
    model = MemoryPolicy(FeedforwardPolicy(1))
    for bad in (torch.zeros(7), torch.full((8,), float("nan"))):
        with pytest.raises(ValueError):
            model(torch.zeros(78), (bad,))


def test_real_actor_tracks_memory_without_changing_zero_initialized_trajectory():
    base = FeedforwardPolicy(3)
    motor = DirectionMotor(5)
    actors = [NeuralForager(base, motor, 7), NeuralForager(MemoryPolicy(base), motor, 7)]
    worlds = [ColonyEnvironment(11, ColonyConfig(ants=1, horizon=40)) for _ in actors]
    for _ in range(40):
        actions = [actor.act(world.observation(0), sampled=True) for actor, world in zip(actors, worlds, strict=True)]
        assert actions[0] == actions[1]
        events = [world.step([action]) for world, action in zip(worlds, actions, strict=True)]
        assert events[0] == events[1]
        np.testing.assert_array_equal(worlds[0].ants[0].position, worlds[1].ants[0].position)
        np.testing.assert_array_equal(worlds[0].signals.trails.values, worlds[1].signals.trails.values)
    assert not actors[0].history and len(actors[1].history) == 16
