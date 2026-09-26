from dataclasses import replace
import gzip
from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.comparison.injury_branch import clone_for_branch, evaluate
from mathhackson.training.comparison.online_actor import OnlineForager
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.revival import RevivingColony
from mathhackson.training.foraging.trust_candidate import TrustConfig
from mathhackson.training.foraging.update_decision import UpdateDecision


def pending(death=False):
    torch.set_num_threads(1)
    actor = OnlineForager(MemoryPolicy(FeedforwardPolicy(71)), DirectionMotor(41), UpdateDecision(),
                         14, 'skip', TrustConfig(window=1, feedback_mode='observed-window'))
    env = RevivingColony(1, ColonyConfig(ants=1, horizon=12),
        DisturbanceConfig(injury_per_step=1. if death else 0., contact_radius=10.))
    action = actor.act(env.observation(0))
    event = env.step([action])[0]
    record = actor.feedback(event.reward, env.observation(0), terminal=event.exhausted,
                            continuing_after_death=event.exhausted, tick=1, individual=0)
    return env, [actor], record


@pytest.mark.parametrize('death', [False, True])
def test_zero_candidate_identical_and_parent_preserved(tmp_path, death):
    env, actors, record = pending(death)
    record = record.model_copy(update={'proposal': replace(record.proposal, delta=(0.,) * 45)})
    rng = actors[0].agent.random.get_state().clone()
    history = [h.detach().clone() for h in actors[0].agent.history]
    position = env.ants[0].position.copy()
    waiting = list(env.waiting)
    result = evaluate(env, actors, record, horizon=3, directory=tmp_path / 'pair')
    assert result.skip == result.accept and not result.changed
    with gzip.open(tmp_path / 'pair/skip.jsonl.gz', 'rb') as a, gzip.open(tmp_path / 'pair/accept.jsonl.gz', 'rb') as b:
        assert a.read() == b.read()
    assert env.steps == 1 and np.array_equal(position, env.ants[0].position)
    assert list(env.waiting) == waiting and torch.equal(rng, actors[0].agent.random.get_state())
    assert all(torch.equal(a, b) for a, b in zip(history, actors[0].agent.history, strict=True))
    if death:
        assert result.skip.focal_revivals == 3 and result.skip.focal_deaths == 3


def test_clone_restores_before_and_changes_only_accept_copy():
    env, actors, record = pending()
    record = record.model_copy(update={'proposal': replace(record.proposal, delta=(.001,) * 45)})
    skip = clone_for_branch(actors, record, accept=False)
    accept = clone_for_branch(actors, record, accept=True)
    np.testing.assert_array_equal(skip[0].agent.weights(), record.before)
    np.testing.assert_allclose(accept[0].agent.weights(), np.asarray(record.before) + .001)
    np.testing.assert_array_equal(actors[0].agent.weights(), record.after)
    assert skip[0].agent.offset.data_ptr() != accept[0].agent.offset.data_ptr()
    record.tick += 1
    with pytest.raises(ValueError):
        evaluate(env, actors, record, horizon=3, directory=Path('unused'))
