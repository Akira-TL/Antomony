"""概率约束候选的逐世界描述；只读已冻结配对数据。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import mean

import numpy as np
from pydantic import BaseModel, ConfigDict

from mathhackson.training.foraging.candidate_probe import CandidateRecord, Execution, WorldRecord
from mathhackson.training.foraging.trust_candidate import TrustConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: str
    manifest: str
    output: str
    tolerance: float = 1e-6


class Difference(BaseModel):
    positive: int
    negative: int
    tied: int
    mean: float | None
    minimum: float | None
    maximum: float | None


class WorldSummary(BaseModel):
    condition: str
    seed: int
    pairs: int
    changed_parameters: int
    reward: Difference
    focal_deliveries: Difference
    colony_deliveries: Difference
    focal_injury: Difference
    focal_death: Difference
    mean_kl: float | None
    maximum_state_kl: float | None


class Summary(BaseModel):
    hashes_verified: int
    unchanged_parent_snapshots: int
    paired_candidates: int
    worlds: list[WorldSummary]


def describe(values: list[float], tolerance: float) -> Difference:
    if not np.isfinite(values).all() or not np.isfinite(tolerance) or tolerance < 0.:
        raise ValueError("差值与数值容差必须有效")
    return Difference(positive=sum(v > tolerance for v in values), negative=sum(v < -tolerance for v in values),
                      tied=sum(abs(v) <= tolerance for v in values), mean=mean(values) if values else None,
                      minimum=min(values) if values else None, maximum=max(values) if values else None)


def summarize(config: Config) -> Summary:
    root = Path(config.input_directory)
    paths: set[Path] = set()
    snapshots = 0
    for line in Path(config.manifest).read_text().splitlines():
        expected, location = line.split("  ", 1)
        path = Path(location)
        if path in paths or not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("原始清单存在重复、越界或散列变化")
        paths.add(path)
        if path.name.endswith(".residual.npz"):
            with np.load(path, allow_pickle=False) as checkpoint:
                if (str(checkpoint["adapter_kind"]) != "direction-trust" or int(checkpoint["writes"]) != 0
                        or np.any(checkpoint["fast"]) or np.any(checkpoint["stable"])):
                    raise ValueError("主轨迹检查点含有未授权参数写入")
            snapshots += 1
    if paths != {p for p in root.rglob("*") if p.is_file()}:
        raise ValueError("原始清单未覆盖完整运行目录")
    execution = Execution.model_validate_json((root / "execution.json").read_text())
    if execution.smoke or execution.plan.candidate_kind != "trust" or not isinstance(execution.plan.adaptation, TrustConfig):
        raise ValueError("输入不是完整概率约束采样")
    for source in execution.sources:
        if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
            raise ValueError("来源模型已改变")
    worlds = [WorldRecord.model_validate_json(line) for line in (root / "worlds.jsonl").read_text().splitlines()]
    expected_keys = {(c, s) for c in execution.plan.conditions for s in execution.plan.seeds}
    if len(worlds) != len(expected_keys) or {(w.condition, w.seed) for w in worlds} != expected_keys:
        raise ValueError("世界身份重复或缺失")
    rows: list[WorldSummary] = []
    for world in worlds:
        pairs = [CandidateRecord.model_validate_json(line) for line in
                 (root / f"{world.condition}-{world.seed}" / "pairs.jsonl").read_text().splitlines()]
        if len(pairs) != world.pairs or len(pairs) > execution.plan.pairs_per_world:
            raise ValueError("配对数量不符")
        if len({(p.tick, p.focal) for p in pairs}) != len(pairs):
            raise ValueError("候选身份重复")
        for pair in pairs:
            if (pair.condition, pair.seed) != (world.condition, world.seed) or not 0 <= pair.focal < execution.plan.environment.ants:
                raise ValueError("配对世界或焦点身份不符")
            if (len(pair.observation), len(pair.hidden), len(pair.proposal.delta)) != (78, 8, 45):
                raise ValueError("观察、隐藏状态或更新维度不符")
            if not np.isfinite([*pair.observation, *pair.hidden, *pair.proposal.delta]).all():
                raise ValueError("非有限候选输入")
            diagnostic = pair.diagnostics
            if (diagnostic is None or not np.isfinite([diagnostic.mean_kl, diagnostic.maximum_kl, diagnostic.surrogate_gain]).all()
                    or not 0. <= diagnostic.mean_kl <= execution.plan.adaptation.maximum_mean_kl + 1e-6
                    or not 0. <= diagnostic.maximum_kl <= execution.plan.adaptation.maximum_state_kl + 1e-6
                    or diagnostic.surrogate_gain <= 0.):
                raise ValueError("非零候选未满足已见概率约束")
            for branch in (pair.result.skip, pair.result.accept):
                if not 0 < branch.steps <= min(execution.plan.branch_horizon, execution.plan.environment.horizon - pair.tick):
                    raise ValueError("分支时限不合法")
                if world.condition == "benign" and branch.colony_deaths:
                    raise ValueError("无害条件出现来源伤害死亡")
        rows.append(WorldSummary(condition=world.condition, seed=world.seed, pairs=len(pairs),
            changed_parameters=sum(p.result.changed for p in pairs),
            reward=describe([p.result.accept.focal_reward - p.result.skip.focal_reward for p in pairs], config.tolerance),
            focal_deliveries=describe([p.result.accept.focal_deliveries - p.result.skip.focal_deliveries for p in pairs], 0.),
            colony_deliveries=describe([p.result.accept.colony_deliveries - p.result.skip.colony_deliveries for p in pairs], 0.),
            focal_injury=describe([p.result.accept.focal_injury - p.result.skip.focal_injury for p in pairs], config.tolerance),
            focal_death=describe([int(p.result.accept.focal_death) - int(p.result.skip.focal_death) for p in pairs], 0.),
            mean_kl=mean(p.diagnostics.mean_kl for p in pairs) if pairs else None,
            maximum_state_kl=max(p.diagnostics.maximum_kl for p in pairs) if pairs else None))
    return Summary(hashes_verified=len(paths), unchanged_parent_snapshots=snapshots,
                   paired_candidates=sum(row.pairs for row in rows), worlds=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_text())
    result = summarize(config)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2) + "\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
