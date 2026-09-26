from dataclasses import replace

import numpy as np
import pytest

from test_update_decision import ready_agent


@pytest.mark.parametrize('accept', [True, False])
def test_preview_matches_actual_commit_without_changing_state(accept):
    agent = ready_agent()
    clone = ready_agent()
    before = agent.weights()
    random = agent.random.get_state().clone()
    history = [h.detach().clone() for h in agent.history]
    rewards = agent.rewards.copy()
    proposal = agent.proposal
    preview = agent.preview_weights(proposal, accept=(accept,))
    np.testing.assert_array_equal(agent.weights(), before)
    assert agent.random.get_state().equal(random)
    assert all(a.equal(b) for a, b in zip(agent.history, history, strict=True))
    assert agent.proposal is proposal and agent.rewards == rewards
    assert agent.writes == agent.decisions == 0
    clone.resolve(clone.proposal, accept=(accept,))
    np.testing.assert_array_equal(preview, clone.weights())


def test_preview_uses_same_residual_projection_and_rejects_stale_proposal():
    agent = ready_agent()
    agent.proposal = replace(agent.proposal, delta=(100.,) * 45)
    proposal = agent.proposal
    preview = agent.preview_weights(proposal, accept=(True,))
    assert np.linalg.norm(preview) <= agent.config.maximum_residual_norm
    agent.resolve(proposal, accept=(True,))
    np.testing.assert_array_equal(preview, agent.weights())
    with pytest.raises(ValueError, match='当前'):
        agent.preview_weights(proposal, accept=(True,))
