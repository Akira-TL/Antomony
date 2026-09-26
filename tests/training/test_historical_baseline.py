import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationConfig, NovelDirectionLearner
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner


def observation():
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 3] = np.linspace(.1, 1., 9)
    return LocalObservation(receptors, False, False, True, 0., (1., 1.))


def learner(rate=.5):
    return NovelDirectionLearner(MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41), 1,
        AdaptationConfig(window=2, gamma=.5, feedback_mode="observed-window", historical_baseline_rate=rate))


def feedback(agent, rewards):
    for reward in rewards:
        agent.act(observation())
        agent.feedback(reward)


def test_current_window_cannot_change_its_own_baseline():
    agent = learner()
    feedback(agent, [2., 4.])
    torch.testing.assert_close(agent.feedback_advantage(observation()), torch.tensor([4., 4.]))
    proposal = agent.propose(observation())
    torch.testing.assert_close(agent.feedback_advantage(observation()), torch.tensor([4., 4.]))
    agent.resolve(proposal, accept=(False,))
    feedback(agent, [-2., 0.])
    # The previous return mean was 4; rate .5 gives the prior baseline 2.
    expected = torch.tensor([-4., -2.])
    torch.testing.assert_close(agent.feedback_advantage(observation()), expected)
    agent.propose(observation())
    torch.testing.assert_close(agent.feedback_advantage(observation()), expected)


def test_saved_baseline_and_revival_preserve_only_own_past(tmp_path):
    agent, other = learner(), learner()
    feedback(agent, [2., 4.])
    agent.resolve(agent.propose(observation()), accept=(False,))
    path = tmp_path / "baseline.npz"
    agent.save(path)
    restored = NovelDirectionLearner.load(path, agent.motor, 1)
    agent.restart(preserve_memory=True)
    for item in (agent, restored, other):
        feedback(item, [-2., 0.])
    for item in (agent, restored):
        torch.testing.assert_close(item.feedback_advantage(observation()), torch.tensor([-4., -2.]))
    torch.testing.assert_close(other.feedback_advantage(observation()), torch.tensor([-2., 0.]))


def test_branch_copies_baseline_without_absorbing_parent_pending_rewards():
    agent = learner()
    feedback(agent, [2., 4.])
    agent.resolve(agent.propose(observation()), accept=(False,))
    feedback(agent, [-2., 0.])
    agent.propose(observation())
    branch = EvaluationActor(agent, accept=True)
    feedback(branch.agent, [-2., 0.])
    torch.testing.assert_close(branch.agent.feedback_advantage(observation()), torch.tensor([-4., -2.]))
    agent.resolve(agent.proposal, accept=(False,))
    feedback(agent, [0., 0.])
    torch.testing.assert_close(agent.feedback_advantage(observation()), torch.tensor([-.5, -.5]))
    torch.testing.assert_close(branch.agent.feedback_advantage(observation()), torch.tensor([-4., -2.]))


def test_baseline_cannot_silently_combine_with_uncalibrated_critic():
    with pytest.raises(ValueError, match="已发生窗口"):
        AdaptationConfig(historical_baseline_rate=.5)


@pytest.mark.parametrize("field,value", [("return_baseline", np.nan), ("baseline_windows", -1),
                                         ("baseline_windows", 1.5), ("return_baseline", [1., 2.])])
def test_corrupt_baseline_snapshot_is_rejected(tmp_path, field, value):
    agent = learner()
    path = tmp_path / "baseline.npz"
    agent.save(path)
    residual = path.with_suffix(".residual.npz")
    with np.load(residual, allow_pickle=False) as source:
        fields = {key: source[key].copy() for key in source.files}
    fields[field] = np.asarray(value)
    np.savez(residual, **fields)
    with pytest.raises(ValueError, match="基线"):
        NovelDirectionLearner.load(path, agent.motor, 1)


def test_constant_action_independent_feedback_stops_spurious_candidate():
    config = TrustConfig(window=1, feedback_mode="observed-window", historical_baseline_rate=1.)
    agent = TrustDirectionLearner(MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41), 1, config)
    feedback(agent, [-2.])
    first = agent.propose(observation())
    assert first.gradient_norm > 0.
    agent.resolve(first, accept=(False,))
    feedback(agent, [-2.])
    second = agent.propose(observation())
    assert second.gradient_norm == 0. and not any(second.delta)


def test_disabled_option_keeps_old_proposals_exact():
    a = learner(0.)
    b = NovelDirectionLearner(MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41), 1,
        AdaptationConfig(window=2, gamma=.5, feedback_mode="observed-window"))
    for rewards in ([2., 4.], [-2., 0.], [0., 0.]):
        feedback(a, rewards)
        feedback(b, rewards)
        x, y = a.propose(observation()), b.propose(observation())
        assert x == y
        a.resolve(x, accept=(True,))
        b.resolve(y, accept=(True,))
        np.testing.assert_array_equal(a.weights(), b.weights())


def test_enabled_snapshot_cannot_silently_forget_baseline(tmp_path):
    agent = learner()
    path = tmp_path / "baseline.npz"
    agent.save(path)
    residual = path.with_suffix(".residual.npz")
    with np.load(residual, allow_pickle=False) as source:
        fields = {key: source[key].copy() for key in source.files if key != "return_baseline"}
    np.savez(residual, **fields)
    with pytest.raises(ValueError, match="缺少状态"):
        NovelDirectionLearner.load(path, agent.motor, 1)
