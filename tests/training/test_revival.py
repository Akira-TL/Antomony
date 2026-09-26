import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.revival import RevivingColony


def stopped(env):
    return [DirectionAction(False, 0., 0.) for _ in env.ants]


def test_larger_colony_has_nonoverlapping_positions_and_waits_inside_nest():
    config = ColonyConfig(ants=32)
    plain = ColonyEnvironment(4, config)
    for i, ant in enumerate(plain.ants):
        assert all(np.linalg.norm(ant.position - other.position) >= .36 for other in plain.ants[:i])
    env = RevivingColony(4, config, DisturbanceConfig(injury_per_step=0.))
    assert sum(not a.exhausted for a in env.ants) == 8
    assert env.pending.sum() == 24
    assert env.release_waiting() == []
    env.ants[0].position[:] = (3., 3.)
    assert env.release_waiting() == []  # 首次出巢不是复活。
    assert not env.ants[8].exhausted
    assert np.linalg.norm(env.ants[8].position) < .65
    assert env.pending.sum() == 23


def test_death_returns_carried_food_without_delivery_and_respawns_next_step():
    env = RevivingColony(4, ColonyConfig(ants=2, horizon=5, stock=3),
                         DisturbanceConfig(contact_radius=20., injury_per_step=1.))
    env.ants[0].carrying = True
    env.ants[0].position[:] = (2., 2.)
    env.stock -= 1
    events = env.step(stopped(env))
    assert events[0].exhausted and not events[0].delivered
    assert env.stock == 3 and not env.ants[0].carrying
    assert env.deaths.tolist() == [1, 1] and env.terminations.tolist() == [1, 1]
    assert not env.done
    revived = env.release_waiting()
    assert revived == [0, 1] and env.revivals.tolist() == [1, 1]
    assert all(not a.exhausted and np.linalg.norm(a.position) < .65 for a in env.ants)
    assert not env.killed.any() and not env.injuries.any()
    assert env.deaths.sum() == 2 and env.total_injury.sum() == 2.
    env.step(stopped(env))
    assert env.deaths.sum() == 4
    assert env.stock + sum(a.carrying for a in env.ants) + sum(a.deliveries for a in env.ants) == 3


def test_budget_exhaustion_restores_body_without_free_reward_or_erasing_counts():
    env = RevivingColony(4, ColonyConfig(ants=1, exploration_steps=1, reserve_steps=1, horizon=4),
                         DisturbanceConfig(injury_per_step=0.))
    env.step(stopped(env))
    event = env.step(stopped(env))[0]
    assert event.exhausted and event.reward == -2.
    assert env.deaths[0] == 0 and env.terminations[0] == 1
    assert env.release_waiting() == [0]
    assert env.ants[0].exploration_left == env.ants[0].reserve_left == 1
    assert env.ants[0].deliveries == env.ants[0].budget_returns == 0
    env.step(stopped(env))
    env.step(stopped(env))
    assert env.done and env.release_waiting() == []


@pytest.mark.parametrize("mode,accepted", [("learned", True), ("always", True), ("skip", False)])
def test_death_feedback_can_learn_but_revival_preserves_individual_state(mode, accepted):
    from test_online_actor import actor, observation
    ant = actor(mode)
    obs = observation()
    for tick in range(1, 5):
        ant.act(obs)
        record = ant.feedback(-float(tick), obs, terminal=tick == 4,
            tick=tick, individual=0, continuing_after_death=tick == 4)
    assert record.terminal and record.continuing_after_death
    assert record.accepted == accepted and record.changed == accepted
    history = [h.detach().clone() for h in ant.agent.history]
    parameters = ant.agent.weights().copy()
    random = ant.agent.random.get_state().clone()
    writes, decisions = ant.agent.writes, ant.agent.decisions
    ant.revive()
    assert not ant.agent.terminal
    assert all(a.equal(b) for a, b in zip(history, ant.agent.history, strict=True))
    np.testing.assert_array_equal(ant.agent.weights(), parameters)
    assert torch.equal(random, ant.agent.random.get_state())
    assert (ant.agent.writes, ant.agent.decisions) == (writes, decisions)
    ant.check_frozen()
    ant.act(obs)
