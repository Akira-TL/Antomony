"""接受学习的固定分支描述；按世界及种子汇总，不拼接为在线轨迹。"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.foraging.update_curriculum import CurriculumPlan, LabeledDecision, world_key
from mathhackson.training.foraging.update_decision import UpdateDecision


class WorldScore(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    key: str
    seed: int
    initial_norm: float
    count: int
    accepted: int
    positive: int
    negative: int
    tied: int
    selected_reward: float | None
    always_reward: float | None
    prediction_mse: float | None
    selected_focal_deliveries: float | None
    selected_colony_deliveries: float | None


class SeedScore(BaseModel):
    seed: int
    complete: bool
    selected_reward: float | None
    always_reward: float | None
    better_than_both: bool


class DecisionSummary(BaseModel):
    seeds: list[SeedScore]
    complete_seeds: int
    passing_seeds: int
    overall_selected: float | None
    overall_always: float | None
    development_continue: bool


def score_world(key: str, seed: int, norm: float, labels: np.ndarray, predictions: np.ndarray,
                focal_deliveries: np.ndarray, colony_deliveries: np.ndarray) -> WorldScore:
    values = (labels, predictions, focal_deliveries, colony_deliveries)
    if labels.ndim != 1 or any(v.shape != labels.shape or not np.isfinite(v).all() for v in values):
        raise ValueError("每条候选必须有对应的有限预测、奖励差与交付差")
    accepted = predictions > 0.
    count = len(labels)
    return WorldScore(key=key, seed=seed, initial_norm=norm, count=count, accepted=int(accepted.sum()),
                      positive=int((labels > 1e-6).sum()), negative=int((labels < -1e-6).sum()),
                      tied=int((np.abs(labels) <= 1e-6).sum()),
                      selected_reward=float((labels * accepted).mean()) if count else None,
                      always_reward=float(labels.mean()) if count else None,
                      prediction_mse=float(np.square(labels - predictions).mean()) if count else None,
                      selected_focal_deliveries=float((focal_deliveries * accepted).mean()) if count else None,
                      selected_colony_deliveries=float((colony_deliveries * accepted).mean()) if count else None)


def summarize(worlds: list[WorldScore], seeds: tuple[int, ...], norms: tuple[float, ...],
              *, minimum_passing: int) -> DecisionSummary:
    expected = {world_key(i, seed) for i in range(len(norms)) for seed in seeds}
    if (len(set(seeds)) != len(seeds) or len(set(norms)) != len(norms) or not norms
            or not 1 <= minimum_passing <= len(seeds)
            or len(worlds) != len(expected) or {w.key for w in worlds} != expected):
        raise ValueError("汇总需要完整且无重复的种子与条件")
    scores = []
    for seed in seeds:
        rows = [next(w for w in worlds if w.key == world_key(i, seed)) for i in range(len(norms))]
        if any(w.seed != seed or w.initial_norm != norm for w, norm in zip(rows, norms, strict=True)):
            raise ValueError("条件身份与世界编号不一致")
        complete = all(w.count > 0 and w.selected_reward is not None and w.always_reward is not None for w in rows)
        selected = float(np.mean([w.selected_reward for w in rows])) if complete else None
        always = float(np.mean([w.always_reward for w in rows])) if complete else None
        scores.append(SeedScore(seed=seed, complete=complete, selected_reward=selected, always_reward=always,
                                better_than_both=complete and selected > max(0., always)))
    complete = sum(s.complete for s in scores)
    passing = sum(s.better_than_both for s in scores)
    overall_selected = float(np.mean([s.selected_reward for s in scores])) if complete == len(seeds) else None
    overall_always = float(np.mean([s.always_reward for s in scores])) if complete == len(seeds) else None
    delivery_improved = any(w.selected_focal_deliveries is not None and w.selected_focal_deliveries > 0. for w in worlds)
    proceed = (complete == len(seeds) and passing >= minimum_passing
               and overall_selected > max(0., overall_always) and delivery_improved)
    return DecisionSummary(seeds=scores, complete_seeds=complete, passing_seeds=passing,
                           overall_selected=overall_selected, overall_always=overall_always,
                           development_continue=proceed)


def evaluate_held(rows: list[LabeledDecision], models: list[UpdateDecision], plan: CurriculumPlan) -> list[WorldScore]:
    allowed = {world_key(i, seed) for i in range(len(plan.initial_norms)) for seed in plan.held_seeds}
    if len(models) != plan.probe.environment.ants or any(row.key not in allowed for row in rows):
        raise ValueError("留出评价只能使用预定留出世界及每只个体对应模型")
    result = []
    for i, norm in enumerate(plan.initial_norms):
        for seed in plan.held_seeds:
            key = world_key(i, seed)
            selected = [row for row in rows if row.key == key]
            with torch.inference_mode():
                predictions = np.asarray([float(models[row.focal](torch.from_numpy(row.features))) for row in selected])
            labels = np.asarray([row.benefit for row in selected])
            focal = np.asarray([row.record.result.accept.focal_deliveries - row.record.result.skip.focal_deliveries for row in selected])
            colony = np.asarray([row.record.result.accept.colony_deliveries - row.record.result.skip.colony_deliveries for row in selected])
            result.append(score_world(key, seed, norm, labels, predictions, focal, colony))
    return result
