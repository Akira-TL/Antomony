import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from test_trust_candidates import observation
from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution
from mathhackson.training.foraging.trust_candidate import CandidateDiagnostics, TrustConfig, TrustDirectionLearner


def make_agent(**changes):
    return TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14,
        TrustConfig(window=4, feedback_mode="observed-window", **changes))


def probability(agent, obs, choice):
    with torch.no_grad():
        return float(direction_distribution(agent.predict(torch.from_numpy(obs.vector()), ())[0]).probs[choice])


def test_reward_after_a_skipped_window_can_reinforce_the_earlier_action():
    old, delayed = make_agent(), make_agent(credit_horizon=16)
    first = observation()
    for agent in (old, delayed):
        action = agent.act(first)
        choice = int((DIRECTIONS @ torch.from_numpy(action.direction)).argmax())
        before = probability(agent, first, choice)
        agent.feedback(0.)
        for _ in range(3):
            agent.act(observation(False))
            agent.feedback(0.)
        empty = agent.propose(observation(False))
        assert not np.any(empty.delta)
        agent.resolve(empty, accept=(False,))
        for tick in range(4):
            agent.act(observation(False))
            agent.feedback(float(tick == 3))
        reward = agent.propose(observation(False))
        assert reward.steps == 4 and agent.decisions == 1
        if agent is old:
            assert reward.gradient_norm == 0. and not np.any(reward.delta)
        else:
            assert reward.gradient_norm > 0. and np.any(reward.delta)
            assert agent.diagnostics.credit_steps == 8
            assert agent.resolve(reward, accept=(True,))
            assert probability(agent, first, choice) > before
            # 写入后较早策略的动作不能继续接收新奖励。
            for _ in range(4):
                agent.act(observation(False))
                agent.feedback(1.)
            after = agent.propose(observation(False))
            assert after.gradient_norm == 0. and agent.diagnostics.credit_steps == 4


def test_skipping_retains_actions_but_never_reuses_processed_rewards():
    agent = make_agent(credit_horizon=16)
    for _ in range(4):
        agent.act(observation())
        agent.feedback(1.)
    first = agent.propose(observation())
    assert first.gradient_norm > 0.
    agent.resolve(first, accept=(False,))
    for _ in range(4):
        agent.act(observation())
        agent.feedback(0.)
    second = agent.propose(observation())
    assert second.gradient_norm == 0. and not np.any(second.delta)
    assert agent.diagnostics.credit_steps == 8 and second.steps == 4


def test_older_than_sixteen_steps_is_not_credited_and_buffers_stay_bounded():
    agent = make_agent(credit_horizon=16)
    for tick in range(1, 33):
        agent.act(observation(tick == 1))
        agent.feedback(float(tick == 32))
        if tick % 4 == 0:
            proposal = agent.propose(observation(False))
            assert agent.diagnostics.credit_steps == min(tick, 16)
            assert not np.any(proposal.delta)
            agent.resolve(proposal, accept=(False,))
    assert agent.decisions == 8 and agent.writes == 0


def test_terminal_short_window_clears_credit_but_revival_keeps_hidden_memory():
    agent = make_agent(credit_horizon=16)
    other = make_agent(credit_horizon=16)
    frozen = [p.detach().clone() for p in [*agent.policy.parameters(), *agent.motor.parameters()]]
    for tick in range(4):
        agent.act(observation(tick == 0))
        agent.feedback(0.)
    agent.resolve(agent.propose(observation(False)), accept=(False,))
    agent.act(observation(False))
    agent.feedback(-1., terminal=True)
    proposal = agent.propose(observation(False))
    assert proposal.steps == 1 and agent.diagnostics.credit_steps == 5 and proposal.gradient_norm > 0.
    agent.resolve(proposal, accept=(False,))
    history = [h.detach().clone() for h in agent.history]
    agent.restart(preserve_memory=True)
    assert all(a.equal(b) for a, b in zip(history, agent.history, strict=True))
    assert all(a.equal(b) for a, b in zip(frozen, [*agent.policy.parameters(), *agent.motor.parameters()], strict=True))
    assert not other.history and not other.weights().any()
    for _ in range(4):
        agent.act(observation(False))
        agent.feedback(-1.)
    assert agent.propose(observation(False)).gradient_norm == 0.
    assert agent.diagnostics.credit_steps == 4


@pytest.mark.parametrize("change", [
    {"credit_horizon": 3}, {"credit_horizon": 65}, {"feedback_mode": "critic"},
    {"feedback_trigger": "negative-feedback"}, {"historical_baseline_rate": .1},
])
def test_incompatible_credit_configurations_fail_closed(change):
    values = {"window": 4, "credit_horizon": 16, "feedback_mode": "observed-window"} | change
    with pytest.raises(ValueError):
        TrustConfig.model_validate(values)


def test_old_serialization_stays_unchanged_and_new_snapshot_keeps_config(tmp_path):
    assert "credit_horizon" not in TrustConfig().model_dump()
    assert "credit_steps" not in TypeAdapter(CandidateDiagnostics).dump_python(CandidateDiagnostics())
    assert TypeAdapter(CandidateDiagnostics).dump_python(CandidateDiagnostics(credit_steps=8))["credit_steps"] == 8
    agent = make_agent(credit_horizon=16)
    agent.save(tmp_path / "model.npz")
    loaded = TrustDirectionLearner.load(tmp_path / "model.npz", agent.motor, 14)
    assert loaded.config.credit_horizon == 16 and loaded.config.window == 4
    np.testing.assert_array_equal(loaded.weights(), agent.weights())


def test_deployment_cannot_reuse_old_acceptance_model_with_new_credit_scope():
    from mathhackson.training.comparison.continuous import Condition, ContinuousPlan
    from mathhackson.training.foraging.disturbance import DisturbanceConfig
    values = {"conditions": [Condition(name="reference", disturbance=DisturbanceConfig())],
              "adaptation": make_agent(credit_horizon=16).config}
    with pytest.raises(ValueError, match="旧接受模型"):
        ContinuousPlan.model_validate(values)
    assert ContinuousPlan.model_validate(values | {"arms": ["skip", "always", "mlp", "rules"]}).adaptation.window == 4
