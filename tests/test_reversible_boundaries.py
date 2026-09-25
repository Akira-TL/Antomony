"""独立提取后补充的公开接口边界测试；不代表学习效果实验。"""
from __future__ import annotations

import numpy as np

from mathhackson.fast_residual import FastResidualParameter, RollbackRequest


def test_aliased_delta_keeps_its_original_rollback_amount() -> None:
    parameter = FastResidualParameter([1.0], recent_capacity=3)
    parameter.add_delta([0.5])

    # 调用时增量为 0.5；输入与内部数组共享存储不能改变这笔修改的身份。
    parameter.add_delta(parameter.fast)
    parameter.rollback(RollbackRequest(index_from_oldest=1, fraction=1.0))

    np.testing.assert_array_equal(parameter.effective, [1.5])
    np.testing.assert_array_equal(parameter.recent[0], [0.5])


def test_invalid_commit_is_rejected_before_any_rollback() -> None:
    parameter = FastResidualParameter([1.0], recent_capacity=3)
    parameter.add_delta([0.5])

    with np.testing.assert_raises(ValueError):
        parameter.commit(
            [0.25],
            rollback=RollbackRequest(index_from_oldest=0, fraction=1.0),
            consolidation_fraction=1.1,
        )

    np.testing.assert_array_equal(parameter.stable, [1.0])
    np.testing.assert_array_equal(parameter.fast, [0.5])
    np.testing.assert_array_equal(parameter.recent[0], [0.5])


def test_commit_snapshots_new_delta_before_consolidating_an_alias() -> None:
    parameter = FastResidualParameter([1.0], recent_capacity=3)
    parameter.add_delta([0.5])
    parameter.commit(parameter.fast, consolidation_fraction=0.5)

    np.testing.assert_array_equal(parameter.stable, [1.25])
    np.testing.assert_array_equal(parameter.fast, [0.75])
    np.testing.assert_array_equal(parameter.recent[-1], [0.5])


def test_selective_rollback_is_not_counterfactual_retraining() -> None:
    parameter = FastResidualParameter([0.0], recent_capacity=2)
    parameter.add_delta([0.5])
    parameter.add_delta([0.25])
    parameter.rollback(RollbackRequest(index_from_oldest=0, fraction=1.0))

    # 两次依赖当前参数的平方误差更新：撤销第一笔后，第二笔仍是原来的 0.25。
    # 若从未发生第一笔学习，仅做一次步长 0.5、目标 1 的更新，结果应为 0.5。
    np.testing.assert_array_equal(parameter.effective, [0.25])
    assert float(parameter.effective[0]) != 0.5


def test_consolidation_preserves_a_small_neural_readout_over_many_inputs() -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        hidden = np.tanh(rng.normal(size=(32, 4))).astype(np.float32)
        parameter = FastResidualParameter(rng.normal(size=(4, 2)), recent_capacity=4)
        for _ in range(4):
            parameter.add_delta(rng.normal(scale=0.05, size=(4, 2)))
        before = np.tanh(hidden @ parameter.effective)

        parameter.consolidate(float(rng.uniform(0.0, 1.0)))

        after = np.tanh(hidden @ parameter.effective)
        np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-6)
