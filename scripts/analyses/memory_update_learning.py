"""按固定协议拟合独立接受模型，完成后才评价留出分支。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.comparison.acceptance_audit import Audit, audit_inputs, audit_pairs
from mathhackson.training.comparison.acceptance_scores import DecisionSummary, WorldScore, evaluate_held, summarize
from mathhackson.training.foraging.update_curriculum import FitRecord, fit_individual, read_partition
from mathhackson.training.foraging.update_decision import UpdateDecision


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: Path
    manifest: Path
    protocol: Path
    output_directory: Path
    minimum_passing: int = 3


class Result(BaseModel):
    audit: Audit
    fits: list[FitRecord]
    worlds: list[WorldScore]
    decision: DecisionSummary


def run(config: Config) -> Result:
    execution, audit = audit_inputs(config.input_directory, config.manifest, config.protocol)
    plan = execution.plan
    training = read_partition(config.input_directory, plan, "train")
    audit_pairs(training, plan)
    config.output_directory.mkdir(parents=True, exist_ok=False)
    fits = [fit_individual(training, focal, plan, config.output_directory / f"ant-{focal:02d}")
            for focal in range(plan.probe.environment.ants)]
    held = read_partition(config.input_directory, plan, "held")
    audit_pairs(held, plan)
    models = [UpdateDecision.load(config.output_directory / f"ant-{focal:02d}" / f"step-{plan.training_steps:04d}.npz").eval()
              for focal in range(plan.probe.environment.ants)]
    worlds = evaluate_held(held, models, plan)
    decision = summarize(worlds, plan.held_seeds, plan.initial_norms, minimum_passing=config.minimum_passing)
    result = Result(audit=audit, fits=fits, worlds=worlds, decision=decision)
    with (config.output_directory / "summary.json").open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(run(Config.model_validate_json(args.config.read_text())).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
