"""首步分支的逐帧审计与放宽分布约束后的解析上界。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from .continuous import Frame
from .first_action import Execution, Outcome, Point


class Contrast(BaseModel):
    current: float
    best_direction: float
    relaxed_mixture_upper: float
    worst_direction: float
    directions: list[float]


class Result(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    seed: int
    tick_before: int
    individual: int
    negative_failures: Contrast
    feedback: Contrast
    injury_by_direction: list[float]
    deliveries_by_direction: list[int]
    feedback_best_directions: list[int]
    minimum_failure_directions: list[int]


def contrast(values: list[float], probabilities: list[float]) -> Contrast:
    q, p = np.asarray(values, np.float64), np.asarray(probabilities, np.float64)
    if q.shape != (16,) or p.shape != (16,) or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('必须有16个有限方向结果和概率')
    if np.any(p < .2 / 16 - 1e-7) or not np.isclose(p.sum(), 1., atol=1e-6):
        raise ValueError('当前方向分布不满足20%均匀混合')
    return Contrast(current=float(p @ q), best_direction=float(q.max()),
        relaxed_mixture_upper=float(.2 * q.mean() + .8 * q.max()), worst_direction=float(q.min()), directions=q.tolist())


def audit_point(directory: Path, expected_steps: int) -> Result:
    point = Point.model_validate_json((directory / 'point.json').read_text())
    if [outcome.direction for outcome in point.outcomes] != list(range(16)):
        raise ValueError('方向必须完整且不重复')
    for saved in point.outcomes:
        with gzip.open(directory / f'direction-{saved.direction:02d}.jsonl.gz', 'rt') as stream:
            trace = [Frame.model_validate_json(line) for line in stream]
        if len(trace) != expected_steps or [frame.tick for frame in trace] != list(range(point.tick_before + 1, point.tick_before + expected_steps + 1)):
            raise ValueError('分支不足32步或时钟不连续，不作完整32步效果解释')
        if any(ant.writes for frame in trace for ant in frame.ants):
            raise ValueError('冻结分支出现写入')
        last = trace[-1].ants
        i = point.individual
        failures = last[i].cumulative_terminations - point.initial_failures[i]
        deaths = last[i].cumulative_deaths - point.initial_deaths[i]
        rebuilt = Outcome(direction=saved.direction, steps=len(trace), failures=failures, deaths=deaths,
            exhaustion=failures - deaths, injury=sum(frame.ants[i].injury_delta for frame in trace),
            deliveries=sum(int(frame.ants[i].delivered) for frame in trace),
            feedback=sum(frame.ants[i].learning_reward for frame in trace),
            colony_failures=sum(ant.cumulative_terminations - old for ant, old in zip(last, point.initial_failures, strict=True)),
            colony_deliveries=sum(int(ant.delivered) for frame in trace for ant in frame.ants))
        for name in Outcome.model_fields:
            if not np.isclose(getattr(rebuilt, name), getattr(saved, name), rtol=0., atol=1e-10):
                raise ValueError(f'逐帧结果与记录不一致: {name}')
    failures = [-float(outcome.failures) for outcome in point.outcomes]
    feedback = [outcome.feedback for outcome in point.outcomes]
    return Result(seed=point.seed, tick_before=point.tick_before, individual=point.individual,
        negative_failures=contrast(failures, point.probabilities), feedback=contrast(feedback, point.probabilities),
        injury_by_direction=[outcome.injury for outcome in point.outcomes],
        deliveries_by_direction=[outcome.deliveries for outcome in point.outcomes],
        feedback_best_directions=[i for i, value in enumerate(feedback) if value == max(feedback)],
        minimum_failure_directions=[i for i, value in enumerate(failures) if value == max(failures)])


class Summary(BaseModel):
    execution_sha256: str
    worlds: int
    points: int
    branch_world_steps: int
    results: list[Result]


def audit(directory: Path) -> Summary:
    raw = (directory / 'execution.json').read_bytes()
    execution = Execution.model_validate_json(raw)
    if not execution.completed_at or tuple(parent.seed for parent in execution.parents) != execution.plan.seeds:
        raise ValueError('采样未完整完成')
    results = []
    for parent in execution.parents:
        if len(parent.points) > 4 or len({i for _, i in parent.points}) != len(parent.points):
            raise ValueError('采样点数量或不同个体限制不符')
        for tick, individual in parent.points:
            result = audit_point(directory / f'seed-{parent.seed}/tick-{tick:04d}-ant-{individual:02d}', 32)
            if (result.seed, result.tick_before, result.individual) != (parent.seed, tick, individual):
                raise ValueError('点身份不符')
            results.append(result)
    return Summary(execution_sha256=hashlib.sha256(raw).hexdigest(), worlds=len(execution.parents),
        points=len(results), branch_world_steps=len(results) * 16 * 32, results=results)
