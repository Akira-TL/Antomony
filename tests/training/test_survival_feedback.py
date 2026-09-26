from dataclasses import replace

import pytest

from mathhackson.training.comparison.continuous import ARMS, Condition, ContinuousPlan, make_actors
from mathhackson.training.comparison.feedback import learning_feedback
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyInteraction
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.revival import RevivingColony


def feedback(event, injury=0., active=True):
    return learning_feedback("survival-v1", event, active=active, injury_delta=injury, injury_limit=1.)


@pytest.mark.parametrize("event", [ColonyInteraction(reward=100., exploration_reward=100.),
    ColonyInteraction(picked_up=True, reward=1.), ColonyInteraction(delivered=True, reward=4.),
    ColonyInteraction(budget_return=True, reward=2.)])
def test_positive_task_rewards_cannot_offset_survival_feedback(event):
    assert feedback(event) == 0.
    assert feedback(event, .08) == pytest.approx(-.08)
    assert feedback(replace(event, exhausted=True), .08) == pytest.approx(-1.08)
    assert learning_feedback("legacy", event, active=True, injury_delta=0., injury_limit=1.) == event.reward


def test_feedback_is_bounded_and_waiting_does_not_repeat_failure():
    event = ColonyInteraction(exhausted=True, reward=-999.)
    assert feedback(event, 100.) == -2.
    assert feedback(event, 100., active=False) == 0.
    assert feedback(event) == -1.


@pytest.mark.parametrize("injury", [-.1, float("nan"), float("inf")])
def test_invalid_injury_is_not_silently_clipped(injury):
    with pytest.raises(ValueError):
        feedback(ColonyInteraction(), injury)


def test_real_death_revival_and_exhaustion_count_once():
    env = RevivingColony(4, ColonyConfig(ants=1, horizon=8, exploration_steps=1, reserve_steps=1),
                        DisturbanceConfig(contact_radius=20., injury_per_step=1., active_until=1))
    actions = [DirectionAction(False, 0., 0.)]
    event = env.step(actions)[0]
    assert env.killed[0] and feedback(event, float(env.injuries[0])) == -2.
    assert env.release_waiting() == [0]
    # 执行器在释放等待队列后采样伤害，不能拿上条生命的累计量相减。
    previous = float(env.injuries[0])
    event = env.step(actions)[0]
    assert previous == 0. and feedback(event, float(env.injuries[0]) - previous) == 0.
    event = env.step(actions)[0]
    assert feedback(event) == -1. and not env.killed[0]
    assert env.terminations[0] == 2 and env.deaths[0] == 1


def test_old_protocol_stays_default_and_old_gate_cannot_use_new_objective():
    conditions = (Condition(name="reference", disturbance=DisturbanceConfig()),)
    legacy = ContinuousPlan(conditions=conditions)
    assert legacy.arms == ARMS and legacy.feedback_profile == "legacy"
    with pytest.raises(ValueError, match="旧接受模型"):
        ContinuousPlan(conditions=conditions, feedback_profile="survival-v1")
    plan = ContinuousPlan(conditions=conditions, feedback_profile="survival-v1", arms=("skip", "always"))
    with pytest.raises(ValueError, match="协议"):
        make_actors(plan, 1, "learned")
    for arms in ((), ("skip", "skip")):
        with pytest.raises(ValueError):
            ContinuousPlan(conditions=conditions, arms=arms)
