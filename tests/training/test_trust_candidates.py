import numpy as np
import pytest
import torch
from pathlib import Path

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner, rotation_probabilities
from mathhackson.training.foraging.candidate_value import EvaluationActor
from mathhackson.training.foraging.adaptation import NovelDirectionLearner


def observation(novel=True):
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 1] = np.linspace(.1, 1., 9)
    if novel:
        receptors[:, 3] = np.linspace(.05, 2., 9)
    return LocalObservation(receptors, False, False, True, .1, (.8, 1.))


def learner():
    return TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14, TrustConfig(window=4))


def complete_window(agent, novel=True):
    for tick in range(4):
        agent.act(observation(novel))
        agent.feedback(float(tick - 2))
    return agent.propose(observation(novel))


def test_candidate_changes_distribution_within_limits_before_any_write():
    agent = learner()
    before = agent.weights()
    item = complete_window(agent)
    assert np.array_equal(before, agent.weights())
    bases, features = torch.stack(agent.base_directions), torch.stack(agent.novel_features)
    old = rotation_probabilities(torch.from_numpy(before), bases, features)
    new = rotation_probabilities(torch.from_numpy(before + np.asarray(item.delta, dtype=np.float32)), bases, features)
    divergence = (old * (old.log() - new.log())).sum(-1)
    assert 0. < float(divergence.mean()) <= agent.config.maximum_mean_kl + 1e-6
    assert float(divergence.max()) <= agent.config.maximum_state_kl + 1e-6
    assert agent.diagnostics.surrogate_gain > 0.
    assert agent.diagnostics.mean_kl == pytest.approx(float(divergence.mean()), abs=1e-6)
    assert agent.resolve(item, accept=(True,))
    assert not agent.base_directions and not agent.novel_features and not agent.action_indices


def test_reject_is_exact_and_individual_models_remain_independent():
    agent, other = learner(), learner()
    before = [p.detach().clone() for p in agent.policy.parameters()]
    motor_before = [p.detach().clone() for p in agent.motor.parameters()]
    item = complete_window(agent)
    assert not agent.resolve(item, accept=(False,)) and not agent.weights().any()
    assert not agent.parameter.recent
    item = complete_window(agent)
    assert agent.resolve(item, accept=(True,))
    assert not other.weights().any()
    assert all(a.equal(b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    assert all(a.equal(b) for a, b in zip(motor_before, agent.motor.parameters(), strict=True))


def test_known_signals_are_neutral_and_checkpoint_kind_is_distinct(tmp_path):
    agent = learner()
    plain = torch.from_numpy(observation(False).vector())
    expected = agent.policy(plain)[0].detach().clone()
    item = complete_window(agent, novel=False)
    assert not np.any(item.delta)
    agent.resolve(item, accept=(True,))
    item = complete_window(agent)
    agent.resolve(item, accept=(True,))
    torch.testing.assert_close(agent.predict(plain, ())[0], expected, rtol=0., atol=0.)
    path = tmp_path / "trust.npz"
    agent.save(path)
    loaded = TrustDirectionLearner.load(path, agent.motor, 14, agent.config)
    np.testing.assert_array_equal(loaded.weights(), agent.weights())
    with pytest.raises(ValueError):
        NovelDirectionLearner.load(path, agent.motor, 14)


def test_failed_calls_do_not_append_buffers_and_terminal_updates_keep_bounds():
    agent = learner()
    agent.act(observation())
    with pytest.raises(ValueError):
        agent.act(observation())
    assert len(agent.base_directions) == 1
    agent.feedback(-2., terminal=True)
    item = agent.propose(observation())
    agent.resolve(item, accept=(True,))
    assert np.linalg.norm(agent.parameter.fast) <= agent.config.maximum_residual_norm
    agent.restart()
    assert not agent.base_directions and not agent.history


def test_branch_accepts_only_clone_and_preserves_parent_random_state():
    agent = learner()
    item = complete_window(agent)
    before = agent.random.get_state().clone()
    accepted, skipped = EvaluationActor(agent, accept=True), EvaluationActor(agent)
    assert accepted.changed and not skipped.changed
    assert np.any(accepted.agent.weights()) and not skipped.agent.weights().any()
    assert not agent.weights().any() and agent.proposal is item
    accepted.act(observation())
    assert agent.random.get_state().equal(before)


def test_probe_preserves_old_plan_and_serializes_new_constraints():
    from mathhackson.training.foraging.candidate_probe import ProbePlan
    old = ProbePlan.model_validate_json(Path(".research/protocols/candidate-update-value.json").read_text())
    new = ProbePlan.model_validate_json(Path(".research/protocols/trust-candidate-value.json").read_text())
    assert old.candidate_kind == "gradient" and not isinstance(old.adaptation, TrustConfig)
    assert new.candidate_kind == "trust" and isinstance(new.adaptation, TrustConfig)
    assert ProbePlan.model_validate_json(new.model_dump_json()) == new
    with pytest.raises(ValueError):
        ProbePlan.model_validate(new.model_dump() | {"candidate_kind": "gradient"})


def test_uniform_penalty_scaling_does_not_increase_normalized_trust_step():
    agents = [TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14,
              TrustConfig(window=16, feedback_mode="observed-window")) for _ in range(2)]
    proposals = []
    for agent, scale in zip(agents, (1., 100.), strict=True):
        for _ in range(16):
            agent.act(observation())
            agent.feedback(-.08 * scale)
        proposals.append(agent.propose(observation()))
    assert np.any(proposals[0].delta)
    assert proposals[1].gradient_norm == pytest.approx(100. * proposals[0].gradient_norm, rel=1e-5)
    # 参数解受病态特征矩阵的舍入影响，比较实际行动分布而非逐参数相等。
    distributions = [rotation_probabilities(torch.tensor(p.delta), torch.stack(a.base_directions),
                     torch.stack(a.novel_features)) for a, p in zip(agents, proposals, strict=True)]
    torch.testing.assert_close(distributions[0], distributions[1], rtol=1e-4, atol=1e-6)
    assert np.linalg.norm(proposals[1].delta) < 1.01 * np.linalg.norm(proposals[0].delta)
    assert agents[0].diagnostics.backtracks == agents[1].diagnostics.backtracks
