import numpy as np
import pytest
import torch

from test_continuous import plan
from mathhackson.training.comparison.acceptance import TapeStore
from mathhackson.training.comparison.continuous import Condition, ContinuousPlan, run
from mathhackson.training.comparison.update_constraints import motor_variation, probability_jacobian, replay_world
from mathhackson.training.comparison.steering_audit import MotorTable
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.trust_candidate import rotation_probabilities


def test_probability_jacobian_matches_independent_central_difference():
    rng = np.random.default_rng(822)
    bases = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
    features = rng.normal(0., .15, (2, 45)).astype(np.float32)
    weights = rng.normal(0., .05, 45).astype(np.float32)
    actual = probability_jacobian(weights.astype(float), bases.astype(float), features.astype(float))
    epsilon = .001
    columns = []
    for i in range(45):
        shift = np.zeros(45, np.float32)
        shift[i] = epsilon
        plus = rotation_probabilities(torch.from_numpy(weights + shift), torch.from_numpy(bases), torch.from_numpy(features))
        minus = rotation_probabilities(torch.from_numpy(weights - shift), torch.from_numpy(bases), torch.from_numpy(features))
        columns.append(((plus - minus) / (2. * epsilon)).numpy())
    np.testing.assert_allclose(actual, np.stack(columns, axis=-1), rtol=.005, atol=.0002)
    assert np.linalg.matrix_rank(actual.reshape(-1, 45)) <= 2
    assert not probability_jacobian(weights, bases, np.zeros_like(features)).any()


def test_actual_motor_distribution_merges_identical_commands():
    table = MotorTable(np.zeros(16), np.zeros(16), np.tile([1., 0.], (16, 1)))
    before = np.zeros((1, 16))
    after = before.copy()
    before[0, 0], after[0, 1] = 1., 1.
    assert motor_variation(before, after, table) == 0.
    table.move[1] = 1.
    assert motor_variation(before, after, table) == 1.


def test_reconstruction_counts_zero_controls_and_checks_saved_constraints(plan, tmp_path):
    condition = Condition(name='moving-danger', disturbance=DisturbanceConfig(
        signal_radius=20., contact_radius=20., injury_per_step=1.))
    plan = ContinuousPlan.model_validate({**plan.model_dump(), 'conditions': (condition,),
        'feedback_profile': 'survival-v1', 'respawn': True, 'arms': ('skip', 'always'),
        'environment': plan.environment.model_copy(update={'horizon': 6}), 'checkpoint_every': 2})
    root = tmp_path / 'run'
    run(plan, root, protocol_sha256='synthetic-test')
    store = TapeStore(root)
    for arm in plan.arms:
        result, changes = replay_world(store, 18999, arm)
        assert result.actions_verified == 12 and result.memory_snapshots_verified == 6
        assert result.writes == len(changes)
        assert result.decisions == result.zero_proposals + result.nonzero_proposals
        if arm == 'skip':
            assert not changes
        else:
            assert changes
            assert all(row.probability_jacobian_rank <= row.effect.steps for row in changes)
            assert all(row.mean_kl_fraction <= 1.0001 for row in changes)
    path = root / 'moving-danger-18999-always/updates.jsonl'
    text = path.read_text()
    from mathhackson.training.comparison.online_actor import UpdateRecord
    rows = [UpdateRecord.model_validate_json(line) for line in text.splitlines()]
    row = next(row for row in rows if row.changed)
    row.diagnostics = row.diagnostics.__class__(mean_kl=1., maximum_kl=1.)
    path.write_text('\n'.join(item.model_dump_json() for item in rows) + '\n')
    with pytest.raises(ValueError, match='分布约束'):
        replay_world(store, 18999, 'always')
