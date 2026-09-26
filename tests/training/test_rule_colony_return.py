import numpy as np
import pytest

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.rules import LocalRuleController


@pytest.mark.parametrize("seed", [9601, 9602, 9603, 9604])
def test_rules_complete_delivery_after_finding_food_in_bounded_field(seed):
    env = ColonyEnvironment(seed, ColonyConfig(trail_profile="bounded-local-v2", nest_signal_strength=1.))
    controls = [LocalRuleController(seed * 32 + i) for i in range(env.config.ants)]
    while not env.done:
        actions = [controller.act(env.observation(i)) if not ant.exhausted else DirectionAction(False, 0., 0.)
                   for i, (controller, ant) in enumerate(zip(controls, env.ants, strict=True))]
        env.step(actions)
    detail = [(a.pickups, a.deliveries, a.exhausted, np.round(a.position, 3).tolist()) for a in env.ants]
    assert sum(a.pickups for a in env.ants) > 0, detail
    assert sum(a.deliveries for a in env.ants) > 0, detail


def test_contact_does_not_override_a_usable_local_gradient():
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[1, 0] = 1.
    receptors[5, 0] = .5
    obs = LocalObservation(receptors, False, True, True, 0., (1., 1.))
    control = LocalRuleController(1)
    for _ in range(30):
        action = control.act(obs)
        assert action.move and action.turn == 0.


def test_signal_acquisition_cancels_uninformed_contact_turn():
    receptors = np.zeros((9, 8), dtype=np.float32)
    control = LocalRuleController(1)
    obs = LocalObservation(receptors, False, True, True, 0., (1., 1.))
    assert not control.act(obs).move
    receptors[1, 0] = 1.
    assert control.act(obs).move
    assert control.escape_left == 0
