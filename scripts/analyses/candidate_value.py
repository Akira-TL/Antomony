"""描述已登记配对分支，不把相关候选当作独立重复。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import mean

import numpy as np
from pydantic import BaseModel, ConfigDict

from mathhackson.training.foraging.candidate_probe import CandidateRecord, Execution, WorldRecord


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: str
    manifest: str
    output: str
    tolerance: float = 1e-6


class WorldSummary(BaseModel):
    condition: str
    seed: int
    candidates: int
    changed_parameters: int
    positive_reward: int
    negative_reward: int
    tied_reward: int
    mean_reward_difference: float | None
    min_reward_difference: float | None
    max_reward_difference: float | None
    focal_delivery_better: int
    focal_delivery_worse: int
    colony_delivery_better: int
    colony_delivery_worse: int
    focal_death_fewer: int
    focal_death_more: int


class Summary(BaseModel):
    input_hashes_verified: bool
    paired_candidates: int
    worlds: list[WorldSummary]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_text())
    root = Path(config.input_directory)
    manifest_paths: set[Path] = set()
    for line in Path(config.manifest).read_text().splitlines():
        expected, location = line.split("  ", 1)
        path = Path(location)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"输入快照改变：{path}")
        manifest_paths.add(path)
    execution = Execution.model_validate_json((root / "execution.json").read_text())
    if execution.smoke:
        raise ValueError("不能将冒烟作为完整采样")
    worlds = [WorldRecord.model_validate_json(line) for line in (root / "worlds.jsonl").read_text().splitlines()]
    expected_keys = {(condition, seed) for condition in execution.plan.conditions for seed in execution.plan.seeds}
    if len(worlds) != len(expected_keys) or {(w.condition, w.seed) for w in worlds} != expected_keys:
        raise ValueError("世界清单重复或不完整")
    rows: list[WorldSummary] = []
    for world in worlds:
        path = root / f"{world.condition}-{world.seed}" / "pairs.jsonl"
        if path not in manifest_paths:
            raise ValueError("配对输入未在冻结清单登记")
        pairs = [CandidateRecord.model_validate_json(line) for line in path.read_text().splitlines()]
        if len(pairs) != world.pairs or len(pairs) > execution.plan.pairs_per_world:
            raise ValueError("候选数量与采样记录不符")
        if len({(p.tick, p.focal) for p in pairs}) != len(pairs):
            raise ValueError("候选身份重复")
        for pair in pairs:
            if (pair.condition, pair.seed) != (world.condition, world.seed):
                raise ValueError("候选世界身份不符")
            if not np.isfinite([*pair.observation, *pair.hidden, *pair.proposal.delta]).all():
                raise ValueError("候选存在非有限输入")
            for branch in (pair.result.skip, pair.result.accept):
                if not 0 < branch.steps <= min(execution.plan.branch_horizon, execution.plan.environment.horizon - pair.tick):
                    raise ValueError("分支时限不合法")
                if world.condition == "benign" and branch.colony_deaths:
                    raise ValueError("无害来源出现死亡")
        delta = [p.result.accept.focal_reward - p.result.skip.focal_reward for p in pairs]
        rows.append(WorldSummary(condition=world.condition, seed=world.seed, candidates=len(pairs),
            changed_parameters=sum(p.result.changed for p in pairs),
            positive_reward=sum(d > config.tolerance for d in delta), negative_reward=sum(d < -config.tolerance for d in delta),
            tied_reward=sum(abs(d) <= config.tolerance for d in delta),
            mean_reward_difference=mean(delta) if delta else None,
            min_reward_difference=min(delta) if delta else None, max_reward_difference=max(delta) if delta else None,
            focal_delivery_better=sum(p.result.accept.focal_deliveries > p.result.skip.focal_deliveries for p in pairs),
            focal_delivery_worse=sum(p.result.accept.focal_deliveries < p.result.skip.focal_deliveries for p in pairs),
            colony_delivery_better=sum(p.result.accept.colony_deliveries > p.result.skip.colony_deliveries for p in pairs),
            colony_delivery_worse=sum(p.result.accept.colony_deliveries < p.result.skip.colony_deliveries for p in pairs),
            focal_death_fewer=sum(p.result.accept.focal_death < p.result.skip.focal_death for p in pairs),
            focal_death_more=sum(p.result.accept.focal_death > p.result.skip.focal_death for p in pairs)))
    result = Summary(input_hashes_verified=True, paired_candidates=sum(r.candidates for r in rows), worlds=rows)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2) + "\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
