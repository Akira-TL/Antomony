"""当前奖励及反馈边界的刻画测试，不是期望保留的奖励设计。"""
import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionAction, DirectionMotor
from mathhackson.training.foraging.colony import ColonyConfig, ColonyEnvironment
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.trust_candidate import TrustConfig, TrustDirectionLearner


def world() -> ColonyEnvironment:
    return ColonyEnvironment(8, ColonyConfig(
        ants=1, horizon=600, food_distance_min=8., food_distance_span=2.,
        trail_profile="bounded-local-v2", nest_signal_strength=1.,
    ))


def test_stationary_bounded_deposit_keeps_positive_exploration_reward():
    env = world()
    start = env.ants[0].position.copy()
    events = [env.step([DirectionAction(False, 0., 0.)])[0] for _ in range(159)]
    np.testing.assert_array_equal(env.ants[0].position, start)
    assert env.signals.trails.values.any()
    assert all(0. < e.exploration_reward <= .005 for e in events)
    assert [e.exploration_reward for e in events] == pytest.approx(
        [events[0].exploration_reward] * 159)
    assert all(not e.picked_up and not e.delivered and not e.budget_return for e in events)
    assert env.step([DirectionAction(False, 0., 0.)])[0].exploration_reward == 0.


def test_three_physical_empty_round_trips_earn_six_return_points_without_food():
    env = world()
    ant = env.ants[0]
    ant.heading = 0.
    events = []

    def step(move: bool = False, turn: float = 0.) -> None:
        events.append(env.step([DirectionAction(move, turn, float(move))])[0])

    # 只发合法行动，不重置场、位置或预算；在巢边等待预算结束再返回。
    for cycle in range(3):
        if cycle:
            for _ in range(18):
                step(turn=1.)
        for _ in range(3):
            step(move=True)
        assert ant.away and np.linalg.norm(ant.position) == pytest.approx(1.04)
        for _ in range(18):
            step(turn=1.)
        for _ in range(ant.exploration_left - 2):
            step()
        for _ in range(3):
            step(move=True)
        assert events[-1].budget_return
        assert ant.exploration_left == 160 and ant.reserve_left == 160

    assert env.steps == 483 and ant.budget_returns == 3
    assert ant.pickups == ant.deliveries == 0 and env.stock == env.config.stock
    assert not any(e.exhausted for e in events)
    assert sum(e.reward - e.exploration_reward for e in events) == pytest.approx(6.)
    assert sum(e.reward for e in events) > 6.


@pytest.mark.parametrize("penalty_step", [16, 17])
def test_penalty_after_resolved_window_cannot_credit_its_actions(penalty_step: int):
    receptors = np.zeros((9, 8), dtype=np.float32)
    receptors[:, 3] = np.linspace(.1, 1., 9)
    obs = LocalObservation(receptors, False, False, True, 0., (1., 1.))
    agent = TrustDirectionLearner(MemoryPolicy(FeedforwardPolicy(1)), DirectionMotor(41), 1,
        TrustConfig(window=16, gamma=.995, feedback_mode="observed-window"))
    for tick in range(1, 17):
        agent.act(obs)
        agent.feedback(-2.08 if tick == penalty_step else .005, terminal=tick == penalty_step)
    first = agent.feedback_advantage(obs).clone()
    # 独立手算：先前动作是否收到负反馈仅因惩罚处在边界内外而改变。
    if penalty_step == 16:
        expected = .005 * sum(.995 ** i for i in range(15)) - 2.08 * .995 ** 15
        assert bool((first < 0.).all())
    else:
        expected = .005 * sum(.995 ** i for i in range(16))
        assert bool((first > 0.).all())
    assert first[0].item() == pytest.approx(expected)
    agent.resolve(agent.propose(obs), accept=(False,))
    assert not agent.rewards and not agent.log_probabilities
    if penalty_step == 17:
        agent.act(obs)
        agent.feedback(-2.08, terminal=True)
        torch.testing.assert_close(agent.feedback_advantage(obs), torch.tensor([-2.08]))
        whole_path_return = expected - 2.08 * .995 ** 16
        assert whole_path_return < 0. < first[0].item()
