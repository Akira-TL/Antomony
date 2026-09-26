"""文献公式的独立算术核对；不代表已在环境接入势函数奖励。"""
import math

import pytest

from mathhackson.training.foraging.reward import discounted_returns


def shaping(potentials: list[float], gamma: float) -> list[float]:
    return [gamma * after - before for before, after in zip(potentials[:-1], potentials[1:], strict=True)]


def total(rewards: list[float], gamma: float) -> float:
    return float(discounted_returns(rewards, 0., gamma)[0])


@pytest.mark.parametrize("gamma", [.995, 1.])
@pytest.mark.parametrize("potentials", [[.3, 1., .8, 0.], [.3, -.8, .9, -.4, 0.]])
def test_complete_trajectory_has_only_start_offset(gamma: float, potentials: list[float]):
    base = [0.] * (len(potentials) - 2) + [4.]
    shaped = [r + f for r, f in zip(base, shaping(potentials, gamma), strict=True)]
    assert total(shaped, gamma) == pytest.approx(total(base, gamma) - .3, abs=1e-6)


def test_grzes_two_terminal_counterexample_and_zero_terminal_correction():
    gamma = .995
    original = [0., 100.]
    uncorrected = [original[0] + gamma * 1000., original[1] + gamma * 10.]
    assert original[0] < original[1] and uncorrected[0] > uncorrected[1]
    corrected = [r + shaping([0., 0.], gamma)[0] for r in original]
    assert corrected == original


def test_clipping_negative_shaping_rewards_creates_cycle_income():
    rewards = shaping([0., 1., 0., 1., 0.], .995)
    assert total(rewards, .995) == pytest.approx(0., abs=1e-7)
    assert total([max(0., r) for r in rewards], .995) > 1.9


def test_shaping_discount_must_match_objective_discount():
    correct = shaping([0., 1., 0.], .995)
    wrong = shaping([0., 1., 0.], 1.)
    assert total(correct, .995) == pytest.approx(0., abs=1e-7)
    assert total(wrong, .995) == pytest.approx(.005)


def test_nonterminal_window_retains_action_dependent_endpoint():
    gamma = .995
    potentials = [i / 16 for i in range(17)]
    window = shaping(potentials, gamma)
    assert total(window, gamma) == pytest.approx(gamma ** 16)
    whole = window + shaping([1., 0.], gamma)
    assert total(whole, gamma) == pytest.approx(0., abs=1e-7)
    # 这里只验证一致的后续价值抵消关系，不向当前模型提供未来真实回报。
    assert float(discounted_returns(window, -1., gamma)[0]) == pytest.approx(0., abs=1e-7)


def test_free_termination_at_update_boundary_changes_continuing_objective():
    gamma = .995
    potentials = [i / 16 for i in range(17)] + [1. - i / 16 for i in range(1, 17)]
    whole = shaping(potentials, gamma)
    assert total(whole, gamma) == pytest.approx(0., abs=1e-7)
    first = shaping(potentials[:16] + [0.], gamma)
    second = shaping(potentials[16:], gamma)
    combined = total(first, gamma) + gamma ** 16 * total(second, gamma)
    assert combined == pytest.approx(-gamma ** 16)


def test_revival_transition_must_not_be_omitted_from_continuing_account():
    gamma = .995
    # 三步分别代表离巢、死亡重生转移、之后的整场结束。
    complete = shaping([0., .4, .6, 0.], gamma)
    assert total(complete, gamma) == pytest.approx(0., abs=1e-7)
    omitted = [complete[0], 0., complete[2]]
    assert total(omitted, gamma) == pytest.approx(-gamma * (gamma * .6 - .4))
    assert not math.isclose(total(omitted, gamma), 0., abs_tol=1e-7)
