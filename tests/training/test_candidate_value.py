import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationConfig, NovelDirectionLearner
from mathhackson.training.foraging.candidate_value import EvaluationActor, evaluate_candidate
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony
from mathhackson.training.foraging.policy import ForagingPolicy


def setup_pending():
    env = DisturbedColony(9401, ColonyConfig(ants=2, horizon=40), DisturbanceConfig(injury_per_step=0.))
    agents = [NovelDirectionLearner(ForagingPolicy(71 + i).with_budgets(), DirectionMotor(41), 14 + i,
                                     AdaptationConfig(window=2)) for i in range(2)]
    for _ in range(2):
        actions = [agent.act(env.observation(i)).action for i, agent in enumerate(agents)]
        events = env.step(actions)
        for i, agent in enumerate(agents):
            agent.feedback(events[i].reward - .1)
    for i, agent in enumerate(agents):
        agent.propose(env.observation(i))
    return env, agents


def test_fork_preserves_inference_random_stream_and_detaches_memory():
    env, agents = setup_pending()
    actor = EvaluationActor(agents[0])
    agents[0].resolve(agents[0].proposal, accept=(False,))
    for _ in range(3):
        obs = env.observation(0)
        assert actor.act(obs) == agents[0].act(obs).action
        agents[0].feedback(0.)
        if agents[0].ready:
            agents[0].resolve(agents[0].propose(obs), accept=(False,))
    assert all(h.grad_fn is None for h in actor.agent.history)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(actor.agent.policy.parameters(), agents[0].policy.parameters()))


def test_zero_candidate_branches_identical_and_parent_untouched():
    env, agents = setup_pending()
    agent = agents[0]
    agent.proposal = replace(agent.proposal, delta=(0.,) * 45)
    before = copy.deepcopy(env)
    random_before = [a.random.get_state().clone() for a in agents]
    histories = [[h.detach().clone() for h in a.history] for a in agents]
    proposals = [a.proposal for a in agents]
    rewards = [a.rewards.copy() for a in agents]
    weights = [a.weights() for a in agents]
    result = evaluate_candidate(env, agents, 0, horizon=8)
    assert not result.changed and result.accept == result.skip
    assert result.skip.steps == 8 and env.steps == before.steps
    assert env.stock == before.stock
    for i, a in enumerate(agents):
        assert a.proposal is proposals[i] and a.rewards == rewards[i]
        assert torch.equal(a.random.get_state(), random_before[i])
        assert np.array_equal(a.weights(), weights[i]) and not a.parameter.recent
        assert np.array_equal(env.observation(i).vector(), before.observation(i).vector())
        for actual, expected in zip(a.history, histories[i], strict=True):
            assert torch.equal(actual, expected)


def test_nonzero_candidate_changes_only_focal_clone_and_keeps_update_bound():
    env, agents = setup_pending()
    agents[0].proposal = replace(agents[0].proposal, delta=(.001,) * 45)
    actor = EvaluationActor(agents[0], accept=True)
    assert actor.changed and np.allclose(actor.agent.weights(), .001)
    assert not agents[0].weights().any() and not agents[1].weights().any()
    result = evaluate_candidate(env, agents, 0, horizon=3)
    assert result.changed and np.allclose(result.accepted_weights, .001)
    assert result == evaluate_candidate(env, agents, 0, horizon=3)


def test_terminal_and_incomplete_candidates_are_not_given_alive_labels():
    env, agents = setup_pending()
    agents[0].terminal = True
    with pytest.raises(ValueError):
        evaluate_candidate(env, agents, 0)
    agents[0].terminal = False
    agents[1].awaiting_feedback = True
    with pytest.raises(ValueError):
        evaluate_candidate(env, agents, 0)
    with pytest.raises(ValueError):
        evaluate_candidate(env, agents, 0, horizon=0)
