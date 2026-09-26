"""同一基础课程中的新旧接受输入对照，不用于生存模型资格授予。"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mathhackson.training.foraging.update_curriculum import FitRecord, fit_individual, read_partition
from mathhackson.training.foraging.update_decision import DecisionProfile, UpdateDecision
from .acceptance_audit import Audit, audit_inputs, audit_pairs
from .acceptance_scores import DecisionSummary, WorldScore, evaluate_held, summarize


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: Path
    manifest: Path
    protocol: Path
    output_directory: Path


class ProfileResult(BaseModel):
    profile: DecisionProfile
    fits: list[FitRecord]
    worlds: list[WorldScore]
    decision: DecisionSummary


class InputComparison(BaseModel):
    selected_differences: list[float | None]
    seeds: list[int]
    development_continue: bool


class Result(BaseModel):
    audit: Audit
    parameters: ProfileResult
    actions: ProfileResult
    comparison: InputComparison


def compare(parameters: ProfileResult, actions: ProfileResult) -> InputComparison:
    if parameters.profile != "parameters" or actions.profile != "actions":
        raise ValueError("输入配置对照顺序错误")
    if len(parameters.worlds) != len(actions.worlds):
        raise ValueError("输入对照需要相同的配对样本")
    for old, new in zip(parameters.worlds, actions.worlds, strict=True):
        if ((old.key, old.seed, old.initial_norm, old.count, old.positive, old.negative, old.tied, old.always_reward)
                != (new.key, new.seed, new.initial_norm, new.count, new.positive, new.negative, new.tied, new.always_reward)):
            raise ValueError("输入对照需要相同的配对样本")
    differences = []
    seeds = []
    for old, new in zip(parameters.decision.seeds, actions.decision.seeds, strict=True):
        if old.seed != new.seed or old.complete != new.complete or old.always_reward != new.always_reward:
            raise ValueError("输入对照种子不匹配")
        seeds.append(new.seed)
        differences.append(new.selected_reward - old.selected_reward if old.complete and new.complete else None)
    proceed = (bool(seeds) and actions.decision.development_continue
               and actions.decision.passing_seeds == len(seeds)
               and all(value is not None and value >= 0. for value in differences)
               and any(value is not None and value > 0. for value in differences))
    return InputComparison(selected_differences=differences, seeds=seeds, development_continue=proceed)


def run(config: Config) -> Result:
    execution, audit = audit_inputs(config.input_directory, config.manifest, config.protocol)
    plan = execution.plan
    if plan.decision_profile != "actions" or not plan.probe.include_action_features or plan.probe.adaptation.window != 4:
        raise ValueError("本对照需要显式记录动作特征的四步课程")
    config.output_directory.mkdir(parents=True, exist_ok=False)
    profiles: tuple[DecisionProfile, ...] = ("parameters", "actions")
    plans = [plan.model_copy(update={"decision_profile": profile}) for profile in profiles]
    fits = []
    # 两种输入都拟合完才读取留出标签，不能按留出结果挑训练参数。
    for current in plans:
        rows = read_partition(config.input_directory, current, "train")
        audit_pairs(rows, current)
        fits.append([fit_individual(rows, focal, current,
            config.output_directory / current.decision_profile / f"ant-{focal:02d}")
            for focal in range(plan.probe.environment.ants)])
    results = []
    for current, fit in zip(plans, fits, strict=True):
        rows = read_partition(config.input_directory, current, "held")
        audit_pairs(rows, current)
        models = [UpdateDecision.load(config.output_directory / current.decision_profile / f"ant-{focal:02d}"
                  / f"step-{plan.training_steps:04d}.npz").eval() for focal in range(plan.probe.environment.ants)]
        worlds = evaluate_held(rows, models, current)
        decision = summarize(worlds, plan.held_seeds, plan.initial_norms, minimum_passing=len(plan.held_seeds))
        results.append(ProfileResult(profile=current.decision_profile, fits=fit, worlds=worlds, decision=decision))
    result = Result(audit=audit, parameters=results[0], actions=results[1], comparison=compare(*results))
    with (config.output_directory / "summary.json").open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result
