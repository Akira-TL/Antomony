import copy
import gzip

import numpy as np
import pytest
import torch

from test_continuous import plan
from mathhackson.training.comparison.continuous import Frame
from mathhackson.training.comparison.first_action import inference_actors, record_point, step
from mathhackson.training.comparison.first_action_audit import audit_point, contrast
from mathhackson.training.foraging.revival import RevivingColony
from mathhackson.training.foraging.reward import DIRECTIONS


def test_override_only_replaces_motor_action_preserving_random_and_memory(plan):
    torch.set_num_threads(1)
    env = RevivingColony(18999, plan.environment, plan.conditions[0].disturbance)
    actors = inference_actors(plan, 18999)
    other, cloned = copy.deepcopy(env), copy.deepcopy(actors)
    normal = step(env, actors)
    changed = step(other, cloned, (0, 3))
    target = cloned[0].agent.motor.decide(DIRECTIONS[3].numpy())
    assert (changed.ants[0].move, changed.ants[0].turn) == (target.move, target.turn)
    assert (normal.ants[1].move, normal.ants[1].turn) == (changed.ants[1].move, changed.ants[1].turn)
    for a, b in zip(actors, cloned, strict=True):
        assert a.agent.random.get_state().equal(b.agent.random.get_state())
        assert all(x.equal(y) for x, y in zip(a.agent.history, b.agent.history, strict=True))
        np.testing.assert_array_equal(a.agent.weights(), b.agent.weights())


def test_all_directions_are_isolated_and_audited_without_double_counting(plan, tmp_path):
    torch.set_num_threads(1)
    danger = plan.conditions[0].disturbance.model_copy(update={'contact_radius': 20., 'injury_per_step': 1.})
    env = RevivingColony(18999, plan.environment, danger)
    actors = inference_actors(plan, 18999)
    destination = tmp_path / 'point'
    point = record_point(env, actors, 18999, 0, 6, destination)
    assert env.steps == 0 and not env.terminations.any()
    assert len(point.outcomes) == 16
    assert all(o.failures == o.deaths and o.exhaustion == 0 and o.failures > 0 for o in point.outcomes)
    audit_point(destination, 6)
    file = destination / 'direction-00.jsonl.gz'
    with gzip.open(file, 'rt') as stream:
        trace = [Frame.model_validate_json(line) for line in stream]
    trace[-1].ants[0].cumulative_deaths += 1
    with gzip.open(file, 'wt') as stream:
        stream.writelines(frame.model_dump_json() + '\n' for frame in trace)
    with pytest.raises(ValueError, match='不一致'):
        audit_point(destination, 6)


def test_relaxed_mixture_bound_is_analytic_and_not_rotational_optimum():
    q = [0.] * 15 + [16.]
    value = contrast(q, [1 / 16] * 16)
    assert value.current == 1. and value.best_direction == 16.
    assert value.relaxed_mixture_upper == 13.
    flat = contrast([3.] * 16, [1 / 16] * 16)
    assert flat.current == flat.best_direction == 3.
    assert flat.relaxed_mixture_upper == pytest.approx(3.)
    with pytest.raises(ValueError, match='20%'):
        contrast(q, [0.] * 15 + [1.])


def test_parent_reconstruction_matches_real_online_frames(plan, tmp_path):
    from mathhackson.training.comparison.acceptance import TapeStore
    from mathhackson.training.comparison.continuous import Condition, run
    from mathhackson.training.comparison.first_action import Plan, replay_parent
    condition = Condition(name='moving-danger', disturbance=plan.conditions[0].disturbance.model_copy(
        update={'contact_radius': 20., 'injury_per_step': .1}))
    config = plan.model_copy(update={'conditions': (condition,), 'respawn': True, 'feedback_profile': 'survival-v1', 'arms': ('always',)})
    original = tmp_path / 'original'
    run(config, original, protocol_sha256='test')
    diagnostic = Plan(source_root=original, manifest=tmp_path/'manifest', manifest_sha256='test',
        protocol_sha256='test', seeds=(18999,), points_per_seed=2, branch_steps=6)
    result = replay_parent(diagnostic, TapeStore(original), 18999, tmp_path/'branches')
    assert result.points == [(0, 0), (0, 1)]
    assert result.steps_verified == 1 and result.actions_verified == 2


def test_relative_manifest_rejects_escape_duplicate_missing_and_unlisted(tmp_path):
    import hashlib
    from mathhackson.training.comparison.first_action_audit import verify_relative_manifest
    root = tmp_path / 'raw'
    root.mkdir()
    target = root / 'data'
    target.write_bytes(b'known')
    sha = hashlib.sha256(b'known').hexdigest()
    manifest = tmp_path / 'manifest'
    line = f'{sha}  data\n'
    manifest.write_text(line)
    assert verify_relative_manifest(root, manifest) == 1
    for contents in (line + line, f'{sha}  ../manifest\n', f'{sha}  missing\n', ''):
        manifest.write_text(contents)
        with pytest.raises(ValueError):
            verify_relative_manifest(root, manifest)
