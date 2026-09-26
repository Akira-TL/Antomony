import numpy as np

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig, DisturbedColony


def test_benign_and_dangerous_source_observations_do_not_reveal_consequence():
    safe = DisturbedColony(2, ColonyConfig(ants=1), DisturbanceConfig(injury_per_step=0.))
    danger = DisturbedColony(2, ColonyConfig(ants=1), DisturbanceConfig(injury_per_step=.2))
    for env in (safe, danger):
        env.ants[0].position = env.source_position.copy()
    np.testing.assert_array_equal(safe.observation(0).vector(), danger.observation(0).vector())
    values = danger.observation(0).receptors
    assert values[:, 0].any() and values[:, 3].any() and not values[:, 4:].any()
    action = [DirectionAction(False, 0., 0.)]
    safe_event, danger_event = safe.step(action)[0], danger.step(action)[0]
    assert abs(safe_event.reward - danger_event.reward - .2) < 1e-6


def test_death_only_terminates_affected_individual_and_cannot_charge_twice():
    env = DisturbedColony(2, ColonyConfig(ants=2), DisturbanceConfig(injury_per_step=1.))
    env.ants[0].position = env.source_position.copy()
    actions = [DirectionAction(False, 0., 0.)] * 2
    events = env.step(actions)
    assert events[0].exhausted and events[0].reward == -3.
    assert env.killed.tolist() == [True, False]
    assert not events[1].exhausted
    assert env.step(actions)[0].reward == 0.
    assert env.injuries[0] == 1.


def test_transient_damage_stops_without_removing_its_local_signal():
    env = DisturbedColony(2, ColonyConfig(ants=1), DisturbanceConfig(injury_per_step=.2, active_until=1))
    env.ants[0].position = env.source_position.copy()
    action = [DirectionAction(False, 0., 0.)]
    assert env.step(action)[0].reward == -.2
    assert env.step(action)[0].reward == 0.
    assert env.observation(0).receptors[:, 3].any()
