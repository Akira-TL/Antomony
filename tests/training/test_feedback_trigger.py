import numpy as np
import pytest

from test_online_actor import actor, observation
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner


def early_actor(mode='always'):
    ant = actor(mode)
    ant.agent.config = TrustConfig(window=16, feedback_mode='observed-window', feedback_trigger='negative-feedback')
    return ant


@pytest.mark.parametrize('mode', ['always', 'skip'])
def test_negative_feedback_proposes_immediately_but_does_not_force_acceptance(mode):
    ant = early_actor(mode)
    obs = observation()
    ant.act(obs)
    record = ant.feedback(-.08, obs, terminal=False, tick=1, individual=0)
    assert record is not None and record.proposal.steps == 1
    assert record.eligible and not record.terminal
    assert record.changed == (mode == 'always')
    assert record.accepted == (mode == 'always')
    assert not ant.agent.rewards and not ant.agent.log_probabilities
    ant.check_frozen()
    ant.act(obs)


def test_early_window_retains_preceding_actions_and_never_reuses_feedback():
    ant = early_actor()
    obs = observation()
    for tick in range(1, 4):
        ant.act(obs)
        record = ant.feedback(-.08 if tick == 3 else 0., obs, terminal=False, tick=tick, individual=0)
        if tick < 3:
            assert record is None
    assert record.proposal.steps == 3
    assert record.proposal.mean_reward == pytest.approx(-.08 / 3)
    ant.act(obs)
    record = ant.feedback(-.08, obs, terminal=False, tick=4, individual=0)
    assert record.proposal.steps == 1 and record.proposal.mean_reward == -.08


def test_zero_feedback_keeps_regular_window_and_is_exactly_rejectable():
    ant = early_actor()
    obs = observation()
    for tick in range(1, 17):
        ant.act(obs)
        record = ant.feedback(0., obs, terminal=False, tick=tick, individual=0)
        assert (record is not None) == (tick == 16)
    assert record.proposal.steps == 16 and not record.changed
    assert not np.any(ant.agent.weights())


def test_direct_learner_cannot_ignore_an_early_decision_and_checkpoint_preserves_trigger(tmp_path):
    ant = early_actor()
    obs = observation()
    ant.act(obs)
    ant.agent.feedback(-.08)
    with pytest.raises(ValueError, match='反馈'):
        ant.act(obs)
    proposal = ant.agent.propose(obs)
    ant.agent.resolve(proposal, accept=(False,))
    ant.save(tmp_path / 'ant.npz')
    loaded = TrustDirectionLearner.load(tmp_path / 'ant.npz', ant.agent.motor, 14)
    assert loaded.config.feedback_trigger == 'negative-feedback'
    assert TrustConfig().feedback_trigger == 'window'
    with pytest.raises(ValueError, match='已发生'):
        TrustConfig(feedback_trigger='negative-feedback')
