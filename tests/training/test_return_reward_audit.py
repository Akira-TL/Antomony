import gzip

import numpy as np
import pytest

from mathhackson.training.comparison.continuous import AntFrame, Condition, ContinuousPlan, Frame, run_world
from mathhackson.training.comparison.return_reward_audit import Pair, Totals, audit_world, behavior_equal, compressed_response, decide, known_reward
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.environment import LocalObservation


def ant() -> AntFrame:
    return AntFrame(observation=[0.] * 78, active=True, move=True, turn=0., position=[0., 0.],
        heading=0., carrying=False, exploration_left=160, reserve_left=160,
        picked_up=False, delivered=False, budget_return=True, exhausted=False,
        killed=False, injury=0., reward=2., writes=0)


def test_delivery_does_not_double_count_return():
    empty = ant()
    assert known_reward(empty, 2., 0., 0, 0, 2., 2.) == 2.
    delivered = empty.model_copy(update={'delivered': True})
    assert known_reward(delivered, 2., 0., 0, 0, 2., 2.) == 4.


@pytest.mark.parametrize('raw', [0., .5266456604003906, 2., 8.])
def test_raw_proposal_concentration_matches_compressed_trace(raw):
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[2, 6] = raw
    observation = LocalObservation(receptors, False, False, False, 0.)
    assert compressed_response(raw) == float(observation.vector()[:72].reshape(9, 8)[:, 3:].max())


def test_death_injury_and_exhaustion_are_distinct_costs():
    dead = ant().model_copy(update={'budget_return': False})
    assert known_reward(dead, 2., .08, 1, 0, 2., 2.) == -2.08
    assert known_reward(dead, 2., 0., 0, 1, 2., 2.) == -2.


def test_behavior_comparison_ignores_reward_and_update_counter():
    original = Frame(tick=1, source_position=[0., 0.], source_active=False, ants=[ant()], food_stock=1)
    other = original.model_copy(deep=True)
    other.ants[0].reward = 0.
    assert behavior_equal(original, other)
    other.ants[0].turn = .1
    assert not behavior_equal(original, other)
    other.ants[0].turn = 0.
    other.ants[0].writes = 1
    assert behavior_equal(original, other)
    other.ants[0].position = [1., 0.]
    assert not behavior_equal(original, other)


def pair(seed: int, delivery: int, death: int) -> Pair:
    return Pair(seed=seed, arm='learned', on=Totals(), off=Totals(), delivery_difference=delivery,
                death_difference=death, exhaustion_difference=0, empty_return_difference=0,
                first_behavior_difference=None)


def test_decision_retains_negative_and_mixed_outcomes():
    assert decide([pair(1, 1, 0), pair(2, 0, -1)], 'learned').worth_further_validation
    result = decide([pair(1, 10, -20), pair(2, -1, 1)], 'learned')
    assert result.mean_delivery_difference == 4.5
    assert not result.worth_further_validation
    assert not decide([pair(1, 0, 0), pair(2, 0, 0)], 'learned').worth_further_validation
    with pytest.raises(ValueError):
        decide([pair(1, 0, 0), pair(1, 0, 0)], 'learned')


def test_real_rule_trace_is_audited_and_unearned_reward_rejected(tmp_path):
    condition = Condition(name='moving-danger', disturbance=DisturbanceConfig(active_from=128))
    plan = ContinuousPlan(seeds=(1,), conditions=(condition,), respawn=True,
                          environment=ColonyConfig(ants=2, horizon=4), checkpoint_every=2)
    path = tmp_path / 'world'
    world = run_world(plan, 1, condition, 'rules', path)
    result = audit_world(path, world, plan)
    assert result.deliveries == world.deliveries and result.reward == pytest.approx(world.reward)
    with gzip.open(path / 'trajectory.jsonl.gz', 'rt') as stream:
        trace = [Frame.model_validate_json(line) for line in stream]
    trace[0].ants[0].reward += 2.
    with gzip.open(path / 'trajectory.jsonl.gz', 'wt') as stream:
        stream.writelines(frame.model_dump_json() + '\n' for frame in trace)
    with pytest.raises(ValueError, match='奖励分解'):
        audit_world(path, world, plan)
