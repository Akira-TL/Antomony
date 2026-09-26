from pathlib import Path

import numpy as np
import pytest

from test_online_actor import actor, observation
from mathhackson.training.comparison.continuous import Condition, ContinuousPlan
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner


ROOT = Path(__file__).resolve().parents[2]


def plan():
    return ContinuousPlan.model_validate_json(
        (ROOT / '.research/protocols/four-frame-development.json').read_bytes())


def test_new_entry_uses_four_steps_without_reinterpreting_old_configuration():
    current = plan()
    assert current.adaptation.window == 4
    assert current.adaptation.feedback_trigger == 'window'
    assert current.feedback_profile == 'survival-v1'
    assert current.arms == ('skip', 'always', 'mlp', 'rules')
    with pytest.raises(ValueError, match='旧接受模型'):
        ContinuousPlan.model_validate({**current.model_dump(), 'arms': ['learned']})
    assert TrustConfig().window == 16
    assert ContinuousPlan(conditions=(Condition(name='reference', disturbance=DisturbanceConfig()),)).adaptation.window == 16
    previous = ContinuousPlan.model_validate_json(
        (ROOT / '.research/protocols/feedback-trigger-window.json').read_bytes())
    assert previous.adaptation.window == 16


@pytest.mark.parametrize('mode', ['skip', 'always'])
def test_negative_feedback_waits_for_four_steps_and_decisions_can_be_skipped(mode, tmp_path):
    ant = actor(mode)
    ant.agent.config = plan().adaptation
    obs = observation()
    records = []
    for tick in range(1, 13):
        before = ant.agent.weights().copy()
        ant.act(obs)
        record = ant.feedback(-.08, obs, terminal=False, tick=tick, individual=0)
        assert (record is not None) == (tick % 4 == 0)
        if record is None:
            np.testing.assert_array_equal(ant.agent.weights(), before)
        else:
            records.append(record)
            assert record.proposal.steps == 4
            assert record.proposal.mean_reward == pytest.approx(-.08)
            assert not ant.agent.rewards and not ant.agent.log_probabilities
            if mode == 'skip':
                assert not record.accepted and not record.changed
    assert [record.tick for record in records] == [4, 8, 12]
    assert len(ant.agent.history) == 12
    assert ant.agent.decisions == 3
    if mode == 'always':
        assert any(record.changed for record in records)
    ant.check_frozen()
    ant.save(tmp_path / 'ant.npz')
    restored = TrustDirectionLearner.load(tmp_path / 'ant.npz', ant.agent.motor, 14)
    assert restored.config.window == 4 and restored.config.feedback_trigger == 'window'
    np.testing.assert_array_equal(restored.weights(), ant.agent.weights())


@pytest.mark.parametrize('continuing', [False, True])
def test_short_terminal_window_is_processed_once_and_revival_restarts_count(continuing):
    ant = actor('skip')
    ant.agent.config = plan().adaptation
    obs = observation()
    for tick in (1, 2):
        ant.act(obs)
        record = ant.feedback(-.08, obs, terminal=tick == 2, tick=tick, individual=0,
                              continuing_after_death=continuing and tick == 2)
        assert (record is not None) == (tick == 2)
    assert record.proposal.steps == 2 and record.terminal and not record.changed
    assert not ant.agent.rewards and ant.agent.decisions == 1
    if continuing:
        ant.revive()
        assert len(ant.agent.history) == 2
        for tick in range(3, 7):
            ant.act(obs)
            record = ant.feedback(-.08, obs, terminal=False, tick=tick, individual=0)
            assert (record is not None) == (tick == 6)
        assert record.proposal.steps == 4 and ant.agent.decisions == 2
