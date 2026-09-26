import numpy as np

from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.environment import ForagingConfig, ForagingEnvironment
from mathhackson.training.foraging.rules import LocalRuleController


def test_nest_signal_is_finite_local_and_uses_only_existing_receptor():
    for env in (ForagingEnvironment(9711, ForagingConfig(nest_signal_strength=1.)),
                ColonyEnvironment(9711, ColonyConfig(nest_signal_strength=1.))):
        env.stock = 0
        points = np.asarray([[0., 0.], [.75, 0.], [1.5, 0.], [3., 0.]], dtype=np.float32)
        actual = env.signals.sample(points, env.sources())
        np.testing.assert_allclose(actual[:, 1], [1., .25, 0., 0.])
        assert not actual[:, [0, 2, 3, 4, 5, 6, 7]].any()
    assert ColonyEnvironment(9711).observation(0).vector().shape == (78,)


def test_missing_option_preserves_old_environment_and_isolated_sources():
    old = ColonyEnvironment(9711)
    new = ColonyEnvironment(9711, ColonyConfig(nest_signal_strength=1.))
    old.stock = new.stock = 0
    assert not old.observation(0).receptors.any()
    assert new.observation(0).receptors[:, 1].any()
    np.testing.assert_array_equal(new.food, old.food)
    assert new.ants[0].heading == old.ants[0].heading
    new.ants[0].position = np.asarray([2.5, 0.], dtype=np.float32)
    assert not new.observation(0).receptors.any()


def test_local_nest_source_completes_carried_straight_trail_return():
    env = ColonyEnvironment(9701, ColonyConfig(ants=1, exploration_steps=30,
                                              reserve_steps=160, nest_signal_strength=1.))
    env.stock = 0
    ant = env.ants[0]
    ant.heading = 0.
    while ant.exploration_left:
        env.step([DirectionAction(True, 0., 1.)])
    ant.carrying = True
    controller = LocalRuleController(9701)
    while not env.done and not ant.budget_returns:
        env.step([controller.act(env.observation(0))])
    assert ant.deliveries == 1 and ant.budget_returns == 1 and not ant.exhausted
