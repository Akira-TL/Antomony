import numpy as np
import pytest
import torch

from mathhackson.training.comparison.actors import NeuralForager
from mathhackson.training.comparison.exploration_course import CoursePlan, checkpoint
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.policy import ForagingPolicy


def test_training_and_evaluation_seeds_must_be_separate():
    with pytest.raises(ValueError):
        CoursePlan(training_seeds=(1, 2), evaluation_seeds=(2,))
    with pytest.raises(ValueError):
        CoursePlan(training_seeds=(1, 1))


def test_checkpoints_keep_individual_models_and_optimizer_without_overwrite(tmp_path):
    models = [ForagingPolicy(i, budget_inputs=True) for i in (1, 2)]
    actors = [NeuralForager(model, DirectionMotor(1), i, training=True) for i, model in enumerate(models)]
    assert actors[0].optimizer is not actors[1].optimizer
    assert actors[0].model.direction.weight.data_ptr() != actors[1].model.direction.weight.data_ptr()
    checkpoint(actors, tmp_path, 0)
    for i, actor in enumerate(actors):
        path = tmp_path / f"episode-0000-ant-{i:02d}.npz"
        loaded = ForagingPolicy.load(path)
        assert all(a.equal(b) for a, b in zip(loaded.parameters(), actor.model.parameters(), strict=True))
        optimizer = torch.load(path.with_suffix(".optimizer.pt"), weights_only=True)
        assert optimizer["param_groups"][0]["lr"] == .0003
        with np.load(path, allow_pickle=False) as data:
            assert int(data["update"]) == 0
    with pytest.raises(FileExistsError):
        checkpoint(actors, tmp_path, 0)
