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
