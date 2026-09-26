import numpy as np
import pytest
import torch

from mathhackson.training.comparison.online_actor import OnlineForager
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner
from mathhackson.training.foraging.update_decision import UpdateDecision


def observation():
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 1] = np.linspace(.1, 1., 9)
    receptors[:, 3] = np.linspace(.05, 2., 9)
    return LocalObservation(receptors, False, False, True, .1, (.8, 1.))


def actor(mode="learned", prediction=1.):
    decision = UpdateDecision()
    with torch.no_grad():
        decision.value.bias.fill_(prediction)
    return OnlineForager(MemoryPolicy(FeedforwardPolicy(71)), DirectionMotor(41), decision, 14,
                         mode, TrustConfig(window=4, feedback_mode="observed-window"))


def window(agent, *, terminal=False):
    obs = observation()
    for tick, reward in enumerate((-2., -1., 0., 1.), 1):
        agent.act(obs)
        record = agent.feedback(reward, obs, terminal=terminal and tick == 4, tick=tick, individual=0)
        if tick < 4:
            assert record is None
    return record


@pytest.mark.parametrize("mode,prediction,accepted", [("learned", 1., True), ("learned", -1., False),
                                                     ("always", -1., True), ("skip", 1., False)])
def test_actual_feedback_updates_only_when_selected(mode, prediction, accepted):
    agent = actor(mode, prediction)
    other = actor(mode, prediction)
    record = window(agent)
    assert record.eligible and record.accepted == accepted and record.changed == accepted
    assert record.before == [0.] * 45
    assert np.any(record.after) == accepted
    assert record.prediction == (prediction if mode == "learned" else None)
    assert len(record.features) == 140 and len(record.proposal.delta) == 45
    assert agent.agent.writes == int(accepted) and agent.agent.decisions == 1
    agent.check_frozen()
    other.check_frozen()
    assert not other.agent.weights().any()
    assert {p.data_ptr() for p in agent.frozen_parameters()}.isdisjoint(p.data_ptr() for p in other.frozen_parameters())
    agent.act(observation())


@pytest.mark.parametrize("mode", ["learned", "skip", "always"])
def test_terminal_window_is_resolved_without_writing(mode):
    agent = actor(mode)
    record = window(agent, terminal=True)
    assert record.terminal and not record.eligible and not record.accepted and not record.changed
    assert agent.agent.proposal is None and not agent.agent.rewards
    with pytest.raises(ValueError):
        agent.act(observation())


def test_snapshot_restores_actual_written_parameters(tmp_path):
    agent = actor("always")
    record = window(agent)
    agent.save(tmp_path / "ant.npz")
    restored = TrustDirectionLearner.load(tmp_path / "ant.npz", agent.agent.motor, 0)
    np.testing.assert_array_equal(restored.weights(), record.after)
    assert restored.writes == 1 and restored.decisions == 1


def test_freeze_check_detects_base_mutation():
    agent = actor()
    with torch.no_grad():
        agent.agent.policy.base.middle.bias.add_(1.)
    with pytest.raises(AssertionError):
        agent.check_frozen()
