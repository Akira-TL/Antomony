import numpy as np
import pytest

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.rules import LocalRuleController


@pytest.mark.parametrize("carrying", [False, True])
@pytest.mark.parametrize("heading", [0., np.pi / 4., np.pi / 2., np.pi])
def test_local_rule_returns_along_its_own_straight_outbound_trail(carrying, heading):
    env = ColonyEnvironment(9701, ColonyConfig(ants=1, exploration_steps=30, reserve_steps=160,
                                              trail_profile="bounded-local-v2", nest_signal_strength=1.))
    env.stock = 0
    ant = env.ants[0]
    ant.heading = heading
    while ant.exploration_left:
        env.step([DirectionAction(True, 0., 1.)])
    assert np.linalg.norm(ant.position) > 4.8
    ant.carrying = carrying
    controller = LocalRuleController(9701)
    while not env.done and not ant.budget_returns:
        env.step([controller.act(env.observation(0))])
    assert ant.budget_returns == 1, (ant.position, ant.heading, ant.exhausted)
