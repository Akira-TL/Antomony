from dataclasses import replace

import numpy as np
import pytest
import torch

from test_online_actor import actor, observation, window
from test_update_decision import ready_agent
from mathhackson.training.direction.environment import MAX_TURN
from mathhackson.training.foraging.action_features import action_effect_features
from mathhackson.training.foraging.reward import DIRECTIONS, direction_distribution
from mathhackson.training.foraging.update_decision import (
    ACTION_INPUT_VERSION, ACTION_INPUT_WIDTH, INPUT_VERSION, INPUT_WIDTH,
    IndividualUpdateController, UpdateDecision, train_decision_step,
)


def inverse_bound(values):
    return values / (1. - np.abs(values))


def test_preview_probabilities_match_real_commit_without_consuming_history_or_random():
    ant = ready_agent()
    obs = observation()
    before = ant.weights()
    random = ant.random.get_state().clone()
    history = [h.detach().clone() for h in ant.history]
    values = action_effect_features(ant, obs)
    assert values.shape == (38,) and np.all(np.abs(values) < 1.)
    raw = inverse_bound(values)
    with torch.no_grad():
        p = direction_distribution(ant.predict(torch.from_numpy(obs.vector()), tuple(ant.history))[0]).probs.numpy()
    np.testing.assert_allclose(raw[:16], p, atol=1e-7)
    np.testing.assert_array_equal(ant.weights(), before)
    assert ant.random.get_state().equal(random)
    assert all(a.equal(b) for a, b in zip(history, ant.history, strict=True))
    ant.resolve(ant.proposal, accept=(True,))
    with torch.no_grad():
        q = direction_distribution(ant.predict(torch.from_numpy(obs.vector()), tuple(ant.history))[0]).probs.numpy()
    np.testing.assert_allclose(raw[16:32], q - p, atol=1e-7)
    moves = np.asarray([ant.motor.decide(d.numpy()).move for d in DIRECTIONS])
    assert raw[32] == pytest.approx(p @ moves, abs=1e-7)
    assert raw[33] == pytest.approx(q @ moves, abs=1e-7)
    turns = np.asarray([ant.motor.decide(d.numpy()).turn for d in DIRECTIONS])
    headings = np.stack((np.cos(turns * MAX_TURN), np.sin(turns * MAX_TURN)), axis=-1)
    np.testing.assert_allclose(raw[34:36], [p @ turns, q @ turns], atol=1e-7)
    np.testing.assert_allclose(raw[36:38], ((q - p) * moves) @ headings, atol=1e-7)


def test_zero_candidate_has_zero_action_change_and_invalid_state_is_rejected():
    ant = ready_agent()
    ant.proposal = replace(ant.proposal, delta=(0.,) * 45)
    raw = inverse_bound(action_effect_features(ant, observation()))
    np.testing.assert_array_equal(raw[16:32], 0.)
    np.testing.assert_array_equal(raw[36:38], 0.)
    ant.resolve(ant.proposal, accept=(False,))
    with pytest.raises(ValueError, match='当前候选'):
        action_effect_features(ant, observation())


@pytest.mark.parametrize('profile,width,version', [('parameters', INPUT_WIDTH, INPUT_VERSION),
                                                  ('actions', ACTION_INPUT_WIDTH, ACTION_INPUT_VERSION)])
def test_checkpoint_preserves_input_semantics_and_training_uses_new_features(tmp_path, profile, width, version):
    model = UpdateDecision(profile)
    assert sum(p.numel() for p in model.parameters()) == width + 1
    x = torch.zeros(4, width)
    x[:, -1] = torch.tensor([-.8, -.4, .4, .8])
    target = x[:, -1] * 2.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.001)
    for _ in range(100):
        train_decision_step(model, optimizer, x, target)
    assert torch.equal(model(x).sign(), target.sign())
    path = tmp_path / 'model.npz'
    model.save(path)
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved['version']) == version
    loaded = UpdateDecision.load(path)
    assert loaded.profile == profile and loaded.input_width == width and loaded.updates == 100
    torch.testing.assert_close(loaded(x), model(x), rtol=0., atol=0.)
    with pytest.raises(ValueError, match='维度'):
        loaded(torch.zeros(width - 1))


@pytest.mark.parametrize('prediction,accepted', [(-1., False), (1., True)])
def test_online_actor_records_action_features_without_future_environment(prediction, accepted, monkeypatch):
    from mathhackson.training.foraging import candidate_value
    def forbidden(*args, **kwargs):
        raise AssertionError('部署不能模拟未来')
    monkeypatch.setattr(candidate_value, 'evaluate_candidate', forbidden)
    model = UpdateDecision('actions')
    with torch.no_grad():
        model.value.bias.fill_(prediction)
    ant = actor()
    ant.controller = IndividualUpdateController(model)
    ant.frozen = [p.detach().clone() for p in ant.frozen_parameters()]
    record = window(ant)
    assert len(record.features) == ACTION_INPUT_WIDTH
    assert record.accepted == record.changed == accepted
    assert ant.agent.decisions == 1
    ant.check_frozen()
    with torch.inference_mode():
        assert float(model(torch.tensor(record.features))) == record.prediction
