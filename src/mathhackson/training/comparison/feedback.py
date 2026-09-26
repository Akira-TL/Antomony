"""部署反馈与环境旧奖励分离；不读取坐标、来源类别或未来结果。"""
from __future__ import annotations

import math
from typing import Literal

from mathhackson.training.foraging.colony import ColonyInteraction

FeedbackProfile = Literal["legacy", "survival-v1"]


def learning_feedback(profile: FeedbackProfile, event: ColonyInteraction, *, active: bool,
                      injury_delta: float, injury_limit: float) -> float:
    if profile not in ("legacy", "survival-v1"):
        raise ValueError("未知学习反馈配置")
    if not math.isfinite(injury_delta) or injury_delta < 0.:
        raise ValueError("单步新增伤害须有限且非负")
    if not math.isfinite(injury_limit) or injury_limit <= 0.:
        raise ValueError("伤害上限须有限且为正")
    if not active:
        return 0.
    if profile == "legacy":
        return event.reward
    # exhausted 是本步终止事件，已经包含危险死亡，不再重复扣除 killed。
    # 这是待验证的学习反馈，并非生存/搬运严格排序的数学保证。
    return -float(event.exhausted) - min(injury_delta / injury_limit, 1.)
