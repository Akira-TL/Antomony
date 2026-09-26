import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationConfig, NovelDirectionLearner, NovelSignalLearner
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.policy import ForagingPolicy


def observation(novel=True):
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 1] = np.linspace(.1, 1., 9)
    if novel:
        receptors[:, 3] = np.linspace(.05, 2., 9)
    return LocalObservation(receptors, False, False, True, .1, (.8, 1.))


def learner():
    return NovelSignalLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14, AdaptationConfig(window=4))


def proposal(agent, novel=True, terminal=False):
    for step in range(4):
        agent.act(observation(novel))
        agent.feedback(float(step - 2), terminal=terminal and step == 3)
    return agent.propose(observation(novel))


def test_rejecting_candidate_does_not_write_parameters_or_residual_history():
    agent = learner()
    before = [p.detach().clone() for p in agent.policy.parameters()]
    item = proposal(agent)
    assert item.gradient_norm > 0. and np.any(item.delta)
    assert all(a.equal(b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    assert not agent.resolve(item, accept=(False, False))
    assert all(a.equal(b) for a, b in zip(before, agent.policy.parameters(), strict=True))
    assert not agent.parameter.fast.any() and not agent.parameter.recent and agent.writes == 0
    with pytest.raises(ValueError):
        agent.resolve(item, accept=(True, True))


def test_update_only_selected_novel_module_and_leave_other_individuals_untouched(tmp_path):
    agent, other = learner(), learner()
    before = {key: value.detach().clone() for key, value in agent.policy.state_dict().items()}
    motor_before = [p.detach().clone() for p in agent.motor.parameters()]
    item = proposal(agent)
    assert agent.resolve(item, accept=(True, False))
    assert agent.policy.novel_signal.weight.any() and not agent.policy.novel_strength.weight.any()
    for key, value in agent.policy.state_dict().items():
        if key != "novel_signal.weight":
            assert value.equal(before[key])
    assert not other.weights().any()
    assert all(a.equal(b) for a, b in zip(motor_before, agent.motor.parameters(), strict=True))
    assert np.linalg.norm(agent.parameter.fast) <= .05 + 1e-6
    agent.save(tmp_path / "agent.npz")
    restored = ForagingPolicy.load(tmp_path / "agent.npz")
    assert restored.novel_signal.weight.equal(agent.policy.novel_signal.weight)


def test_no_novel_signal_means_no_update_and_known_inputs_keep_original_function():
    agent = learner()
    plain = torch.from_numpy(observation(False).vector())
    expected = agent.policy(plain)[0].detach().clone()
    zero = proposal(agent, novel=False)
    assert not np.any(zero.delta)
    assert not agent.resolve(zero, accept=(True, True))
    item = proposal(agent)
    agent.resolve(item, accept=(True, True))
    torch.testing.assert_close(agent.policy(plain)[0], expected, rtol=0., atol=0.)


def test_causal_order_terminal_inheritance_and_cross_ant_proposals():
    agent, other = learner(), learner()
    with pytest.raises(ValueError):
        agent.feedback(1.)
    agent.act(observation())
    with pytest.raises(ValueError):
        agent.propose(observation())
    with pytest.raises(ValueError):
        agent.act(observation())
    agent.feedback(-2., terminal=True)
    item = agent.propose(observation())
    with pytest.raises(ValueError):
        other.resolve(item, accept=(True, True))
    agent.resolve(item, accept=(True, True))
    before = agent.weights()
    with pytest.raises(ValueError):
        agent.act(observation())
    agent.restart()
    assert not agent.history and np.array_equal(agent.weights(), before)
    agent.act(observation())


def test_repeated_accepted_updates_keep_total_residual_bounded():
    agent = NovelSignalLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14,
                              AdaptationConfig(window=4, learning_rate=10., maximum_step_norm=.02,
                                               maximum_residual_norm=.03, recent_capacity=2))
    for _ in range(12):
        item = proposal(agent)
        before = agent.weights()
        agent.resolve(item, accept=(True, True))
        assert np.linalg.norm(agent.weights() - before) <= .02 + 1e-6
        assert np.linalg.norm(agent.parameter.fast) <= .03 + 1e-6
        assert len(agent.parameter.recent) <= 2


def test_direction_adapter_starts_neutral_and_updates_no_foundation_weights(tmp_path):
    model, motor = ForagingPolicy(71).with_budgets(), DirectionMotor(41)
    agent = NovelDirectionLearner(model, motor, 14, AdaptationConfig(window=4))
    tensor = torch.from_numpy(observation().vector())
    torch.testing.assert_close(agent.predict(tensor, ())[0], model(tensor)[0], rtol=0., atol=0.)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    item = proposal(agent)
    assert len(item.delta) == 45 and agent.resolve(item, accept=(True,))
    assert not torch.allclose(agent.predict(tensor, ())[0], model(tensor)[0])
    plain = torch.from_numpy(observation(False).vector())
    torch.testing.assert_close(agent.predict(plain, ())[0], model(plain)[0], rtol=0., atol=0.)
    assert all(value.equal(agent.policy.state_dict()[key]) for key, value in initial.items())
    path = tmp_path / "direction.npz"
    agent.save(path)
    loaded = NovelDirectionLearner.load(path, motor, 14, AdaptationConfig(window=4))
    torch.testing.assert_close(loaded.predict(tensor, ())[0], agent.predict(tensor, ())[0], rtol=0., atol=0.)
    assert np.array_equal(loaded.weights(), agent.weights())
