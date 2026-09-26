import math

import numpy as np
import pytest

from test_continuous import plan
from mathhackson.training.comparison.acceptance import TapeStore
from mathhackson.training.comparison.continuous import Condition, ContinuousPlan, run
from mathhackson.training.comparison.steering_audit import MotorTable, audit_world_steering, expected_effect, local_away
from mathhackson.training.direction.environment import STEP_DISTANCE
from mathhackson.training.foraging.disturbance import DisturbanceConfig


def table():
    turn = np.zeros(16)
    turn[4] = math.pi / 18.
    move = np.zeros(16)
    move[0] = 1.
    return MotorTable(move, turn, np.stack((np.cos(turn), np.sin(turn)), axis=-1))


def test_expected_motor_effect_averages_all_sampled_actions():
    p = np.zeros(16)
    p[0] = p[4] = .5
    result = expected_effect(p, table(), np.asarray([0., 1.]))
    assert result.move_probability == .5
    assert result.mean_turn_degrees == pytest.approx(5.)
    assert result.desired_away == pytest.approx(.5)
    assert result.heading_away == pytest.approx(.5 * math.sin(math.pi / 18.))
    assert result.motion_away == 0.
    forward = expected_effect(p, table(), np.asarray([1., 0.]))
    assert forward.motion_away == pytest.approx(.5 * STEP_DISTANCE)


def test_local_gradient_is_body_relative_and_absent_gradient_is_not_zero_effect():
    responses = np.zeros((9, 8))
    responses[1, 6] = 3.
    responses[3, 6] = 1.
    observation = (np.log1p(responses) / math.log(9.)).flatten()
    np.testing.assert_allclose(local_away(observation), [-1., 0.])
    assert local_away(np.zeros(78)) is None
    result = expected_effect(np.full(16, 1. / 16), table(), None)
    assert result.heading_away is None and result.desired_away is None and result.motion_away is None


@pytest.mark.parametrize('p', [np.zeros(16), np.ones(15), np.full(16, float('nan'))])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError, match='概率'):
        expected_effect(p, table(), None)


def test_actual_random_actions_memory_and_death_inheritance_reconstruct(plan, tmp_path):
    condition = Condition(name='moving-danger', disturbance=DisturbanceConfig(
        signal_radius=20., contact_radius=20., injury_per_step=1.))
    plan = ContinuousPlan.model_validate({**plan.model_dump(), 'conditions': (condition,),
        'feedback_profile': 'survival-v1', 'respawn': True, 'arms': ('skip', 'always'),
        'environment': plan.environment.model_copy(update={'horizon': 6}), 'checkpoint_every': 2})
    root = tmp_path / 'run'
    run(plan, root, protocol_sha256='synthetic-test')
    store = TapeStore(root)
    for arm in plan.arms:
        result, changes = audit_world_steering(store, 18999, arm)
        assert result.active_actions_verified == 12 and result.memory_snapshots_verified == 6
        assert result.writes == len(changes)
        if arm == 'always':
            assert result.injured_writes > 0
        else:
            assert result.writes == 0
    path = root / 'moving-danger-18999-always/tick-0002-ant-00.memory.npz'
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    values['random_state'][0] ^= np.uint8(1)
    np.savez(path, **values)
    with pytest.raises(AssertionError):
        audit_world_steering(store, 18999, 'always')
