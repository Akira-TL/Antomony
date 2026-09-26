from dataclasses import replace

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.adaptation import AdaptationProposal
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.trust_candidate import CandidateDiagnostics, TrustConfig, TrustDirectionLearner
from mathhackson.training.foraging.update_decision import (
    INPUT_WIDTH, IndividualUpdateController, UpdateDecision, decision_features, train_decision_step,
)


def observation():
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 1] = np.linspace(.1, 1., 9)
    receptors[:, 3] = np.linspace(.05, 2., 9)
    return LocalObservation(receptors, False, False, True, .1, (.8, 1.))


def ready_agent(*, terminal=False):
    agent = TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14, TrustConfig(window=4))
    for tick in range(4):
        agent.act(observation())
        agent.feedback(float(tick - 2), terminal=terminal and tick == 3)
    agent.propose(observation())
    return agent


def test_features_have_only_explicit_present_inputs_and_fixed_bounds():
    item = AdaptationProposal(1, (0.,) * 45, 2., -.1, -.2, .3, 16)
    inputs = np.arange(78, dtype=np.float32)
    features = decision_features(inputs, np.zeros(8), item, CandidateDiagnostics())
    assert features.shape == (INPUT_WIDTH,) and np.isfinite(features).all()
    assert np.all(np.abs(features) < 1.)
    np.testing.assert_allclose(features[:78], inputs / (1. + inputs))
    # 提案编号也是运行身份，不属于学习输入。
    np.testing.assert_array_equal(features, decision_features(inputs, np.zeros(8), replace(item, number=999), CandidateDiagnostics()))
    with pytest.raises(ValueError):
        decision_features(inputs[:76], np.zeros(8), item, CandidateDiagnostics())
    with pytest.raises(ValueError):
        decision_features(inputs, np.zeros(8), replace(item, mean_reward=float("nan")), CandidateDiagnostics())


def test_synthetic_regression_learns_both_signs_and_saves_dynamic_parameters(tmp_path):
    model = UpdateDecision()
    assert sum(p.numel() for p in model.parameters()) == 141
    before = tmp_path / "before.npz"
    model.save(before)
    features = torch.zeros(4, INPUT_WIDTH)
    features[:, 0] = torch.tensor([-1., -.5, .5, 1.])
    labels = features[:, 0] * 2.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.001)
    losses = [train_decision_step(model, optimizer, features, labels) for _ in range(120)]
    assert losses[-1] < losses[0] * .001
    assert torch.equal(model(features).sign(), labels.sign())
    after = tmp_path / "after.npz"
    model.save(after)
    loaded = UpdateDecision.load(after)
    assert loaded.updates == 120
    torch.testing.assert_close(loaded(features), model(features), rtol=0., atol=0.)
    assert not torch.any(UpdateDecision.load(before)(features))
    with pytest.raises(FileExistsError):
        model.save(after)


def test_decision_rejection_is_exact_and_acceptance_changes_only_own_residual():
    source = UpdateDecision()
    left, right = IndividualUpdateController(source), IndividualUpdateController(source)
    agent, other = ready_agent(), ready_agent()
    random_before = agent.random.get_state().clone()
    assert not left.resolve(agent, observation()).accepted
    assert not np.any(agent.weights()) and agent.writes == 0
    assert agent.random.get_state().equal(random_before)
    frozen = [p.detach().clone() for p in other.policy.parameters()]
    motor = [p.detach().clone() for p in other.motor.parameters()]
    with torch.no_grad():
        right.model.value.bias.fill_(1.)
    choice = right.resolve(other, observation())
    assert choice.eligible and choice.accepted and choice.changed
    assert np.any(other.weights()) and not np.any(agent.weights())
    assert left.model.value.bias.item() == source.value.bias.item() == 0.
    assert all(torch.equal(a, b) for a, b in zip(frozen, other.policy.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(motor, other.motor.parameters(), strict=True))
    with pytest.raises(ValueError):
        right.resolve(other, observation())


def test_terminal_skips_even_with_positive_prediction_and_bad_prediction_does_not_write():
    model = UpdateDecision()
    with torch.no_grad():
        model.value.bias.fill_(1.)
    controller = IndividualUpdateController(model)
    terminal = ready_agent(terminal=True)
    choice = controller.resolve(terminal, observation())
    assert not choice.eligible and not choice.accepted and not choice.changed
    agent = ready_agent()
    proposal = agent.proposal
    with torch.no_grad():
        controller.model.value.bias.fill_(float("nan"))
    with pytest.raises(ValueError):
        controller.resolve(agent, observation())
    assert agent.proposal is proposal and not np.any(agent.weights())


def test_training_rejects_wrong_optimizer_and_nonfinite_or_empty_labels():
    model, other = UpdateDecision(), UpdateDecision()
    optimizer = torch.optim.AdamW(other.parameters())
    x = torch.zeros(2, INPUT_WIDTH)
    with pytest.raises(ValueError):
        train_decision_step(model, optimizer, x, torch.zeros(2))
    optimizer = torch.optim.AdamW(model.parameters())
    for features, target in ((x, torch.tensor([0., float("nan")])), (x[:0], torch.zeros(0))):
        with pytest.raises(ValueError):
            train_decision_step(model, optimizer, features, target)
    assert model.updates == 0


def test_offline_training_does_not_backpropagate_into_action_model_or_label():
    model = UpdateDecision()
    optimizer = torch.optim.AdamW(model.parameters())
    features = torch.ones(2, INPUT_WIDTH, requires_grad=True)
    targets = torch.ones(2, requires_grad=True)
    train_decision_step(model, optimizer, features, targets)
    assert features.grad is None and targets.grad is None


def test_real_world_decisions_need_no_future_simulator(monkeypatch):
    from mathhackson.training.direction.policy import DirectionAction
    from mathhackson.training.foraging import candidate_value
    from mathhackson.training.foraging.colony import ColonyConfig
    from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony

    def forbidden(*args, **kwargs):
        raise AssertionError("部署不能调用未来分支")

    monkeypatch.setattr(candidate_value, "evaluate_candidate", forbidden)
    env = DisturbedColony(71, ColonyConfig(ants=1, horizon=12), DisturbanceConfig(injury_per_step=0.))
    agent = TrustDirectionLearner(ForagingPolicy(71).with_budgets(), DirectionMotor(41), 14, TrustConfig(window=4))
    controller = IndividualUpdateController(UpdateDecision())
    decisions = 0
    while not env.done:
        action = agent.act(env.observation(0)).action if not env.ants[0].exhausted else DirectionAction(False, 0., 0.)
        event = env.step([action])[0]
        agent.feedback(event.reward, terminal=env.done or event.exhausted)
        if agent.ready:
            agent.propose(env.observation(0))
            assert not controller.resolve(agent, env.observation(0)).accepted
            decisions += 1
    assert decisions == 3 and agent.writes == 0
