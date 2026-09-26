"""按预定窗口和个体配对汇总；不合并独立初始化作总体优势推断。"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..feedback_cues import CONDITIONS, Condition
from .rollout import Mode
from .run import EvaluationRow


class Contrast(BaseModel):
    condition: Condition
    comparator: Mode
    episodes: int
    mean_difference: float
    last_quarter_difference: float


class SeedAssessment(BaseModel):
    initialization: int
    contrasts: list[Contrast]
    accepted_fraction: float
    nonzero_writes: int
    association_gain: bool
    reference_preserved: bool
    learned_timing_gain: bool
    mixed_control: bool
    passed: bool


def assess(rows: list[EvaluationRow], *, initialization: int, steps: int) -> SeedAssessment:
    selected = [row for row in rows if row.initialization == initialization]
    groups: dict[tuple[Condition, Mode], list[EvaluationRow]] = {}
    for row in selected:
        groups.setdefault((row.condition, row.mode), []).append(row)
    contrasts = []
    for condition in CONDITIONS:
        learned = sorted(groups.get((condition, "learned"), []), key=lambda row: row.batch_seed)
        if not learned:
            raise ValueError("缺少学习型接受条件")
        for mode in ("off", "always", "matched"):
            compared = sorted(groups.get((condition, mode), []), key=lambda row: row.batch_seed)
            if [row.batch_seed for row in learned] != [row.batch_seed for row in compared]:
                raise ValueError("对照课程种子不匹配")
            differences, last = [], []
            for first, second in zip(learned, compared, strict=True):
                if len(first.mean_after_feedback) != len(second.mean_after_feedback):
                    raise ValueError("对照回合数不匹配")
                differences.extend(np.asarray(first.mean_after_feedback) - np.asarray(second.mean_after_feedback))
                last.extend(np.asarray(first.mean_last_quarter) - np.asarray(second.mean_last_quarter))
            contrasts.append(Contrast(condition=condition, comparator=mode, episodes=len(differences),
                mean_difference=float(np.mean(differences)), last_quarter_difference=float(np.mean(last))))
    values = {(row.condition, row.comparator): row for row in contrasts}
    associated = values[("association", "off")]
    association_gain = associated.mean_difference >= .03 and associated.last_quarter_difference >= .03
    reference = values[("reference", "off")]
    reference_preserved = reference.mean_difference >= -.02 and reference.last_quarter_difference >= -.02
    timing = all(np.mean([values[(condition, mode)].mean_difference for condition in ("association", "transient")]) >= .005
        for mode in ("always", "matched"))
    applicable = [row for row in selected if row.mode == "learned" and row.condition in ("association", "transient")]
    count = sum(sum(row.acceptances) for row in applicable)
    opportunities = sum(len(row.acceptances) * (steps // 4) for row in applicable)
    fraction = count / opportunities
    writes = sum(sum(row.nonzero_writes) for row in applicable)
    mixed = .1 <= fraction <= .9 and writes > 0
    return SeedAssessment(initialization=initialization, contrasts=contrasts, accepted_fraction=fraction,
        nonzero_writes=writes, association_gain=association_gain, reference_preserved=reference_preserved,
        learned_timing_gain=bool(timing), mixed_control=mixed,
        passed=bool(association_gain and reference_preserved and timing and mixed))
