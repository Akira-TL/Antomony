import numpy as np
import pytest
import torch

from mathhackson.training.foraging.feedback_cues import CueConfig, cue_episode
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.plastic_course.run import Initialization, Protocol, Source, base_directions, save_trace
from mathhackson.training.foraging.plastic_course.rollout import rollout
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.direction.policy import DirectionMotor


def protocol():
    source = Source(path="unused.npz", sha256="a" * 64)
    return Protocol(motor=source, initializations=(
        Initialization(seed=1701, foundation=source, train_seed=510000, evaluation_seed=620000),
        Initialization(seed=1702, foundation=source, train_seed=530000, evaluation_seed=640000)))


def test_partitions_and_numeric_limits_are_checked_before_training():
    plan = protocol()
    assert plan.checkpoint_interval == 25
    overlap = plan.initializations[0].model_copy(update={"evaluation_seed": 510020})
    with pytest.raises(ValueError, match="不可重叠"):
        Protocol(**(plan.model_dump() | {"initializations": (overlap, plan.initializations[1])}))
    with pytest.raises(ValueError, match="两个不同"):
        Protocol(**(plan.model_dump() | {"initializations": (plan.initializations[0],)}))
    with pytest.raises(ValueError, match="上限"):
        Protocol(**(plan.model_dump() | {"max_step": 2., "max_fast": .1}))


def test_trace_keeps_every_evaluation_frame_and_frozen_base(tmp_path):
    episode = cue_episode(908901, CueConfig(steps=8, batch=2))
    base = MemoryPolicy(FeedforwardPolicy(10))
    before = {key: value.clone() for key, value in base.state_dict().items()}
    directions = base_directions(base, episode)
    assert not directions.requires_grad
    model = PlasticDirection(DirectionMotor(12), seed=13)
    with torch.no_grad():
        trace = rollout(model, episode, directions, seed=908902)
    path = tmp_path / "trace.npz"
    save_trace(path, episode, directions, trace, structure_update=0)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["observations"].shape == (8, 2, 78)
        assert int(archive["structure_update"]) == 0
        assert archive["fast"].shape == (8, 2, 45, 2)
        assert archive["hidden"].shape == (8, 2, 8)
        assert np.array_equal(archive["rewards"], trace.rewards.numpy())
    with pytest.raises(FileExistsError):
        save_trace(path, episode, directions, trace, structure_update=0)
    assert all(torch.equal(value, before[key]) for key, value in base.state_dict().items())
