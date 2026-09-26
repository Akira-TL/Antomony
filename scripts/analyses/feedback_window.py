"""共同状态下比较两种反馈范围；匹配失败时拒绝解释结果。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from mathhackson.training.foraging.candidate_probe import CandidateRecord, Execution, WorldRecord


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: Path
    manifest: Path
    output: Path


class Difference(BaseModel):
    positive: int
    negative: int
    tied: int
    mean: float | None
    minimum: float | None
    maximum: float | None


def describe(values: list[float], tolerance: float = 1e-6) -> Difference:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("差值必须有限")
    return Difference(positive=int((array > tolerance).sum()), negative=int((array < -tolerance).sum()),
                      tied=int((np.abs(array) <= tolerance).sum()), mean=float(array.mean()) if values else None,
                      minimum=float(array.min()) if values else None, maximum=float(array.max()) if values else None)


class WindowResult(BaseModel):
    window: int
    reward: Difference
    focal_deliveries: Difference
    colony_deliveries: Difference
    zero_feedback: int
    zero_candidates: int
    step_norm: Difference


class WorldResult(BaseModel):
    seed: int
    pairs: int
    windows: list[WindowResult]
    long_minus_short_reward: Difference
    long_minus_short_focal_deliveries: Difference
    long_minus_short_colony_deliveries: Difference


class Result(BaseModel):
    files_verified: int
    snapshots_verified: int
    worlds: list[WorldResult]
    select_long_window: bool


def summarize(window: int, rows: list[CandidateRecord]) -> WindowResult:
    return WindowResult(window=window,
        reward=describe([r.result.accept.focal_reward - r.result.skip.focal_reward for r in rows]),
        focal_deliveries=describe([float(r.result.accept.focal_deliveries - r.result.skip.focal_deliveries) for r in rows], 0.),
        colony_deliveries=describe([float(r.result.accept.colony_deliveries - r.result.skip.colony_deliveries) for r in rows], 0.),
        zero_feedback=sum(r.proposal.mean_reward == r.proposal.minimum_reward == 0. for r in rows),
        zero_candidates=sum(not np.any(r.proposal.delta) for r in rows),
        step_norm=describe([float(np.linalg.norm(r.proposal.delta)) for r in rows]))


def verify_files(config: Config) -> int:
    seen: set[Path] = set()
    for line in config.manifest.read_text().splitlines():
        expected, raw = line.split("  ", 1)
        path = Path(raw)
        if path in seen or not path.resolve().is_relative_to(config.input_directory.resolve()):
            raise ValueError("清单重复或超出当前数据集")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("文件散列不符")
        seen.add(path)
    if seen != {p for p in config.input_directory.rglob("*") if p.is_file()}:
        raise ValueError("清单未完整覆盖数据集")
    return len(seen)


def audit(directory: Path, execution: Execution, world: WorldRecord) -> tuple[list[CandidateRecord], int]:
    plan = execution.plan
    snapshots = 0
    for focal in range(plan.environment.ants):
        rng = np.random.default_rng(world.seed * 8 + focal)
        initial = np.zeros((9, 5), dtype=np.float32)
        initial[:, 0] = rng.choice((-1., 1.)) * .25
        ticks = set(range(64, world.steps + 1, 64)) | {0, world.steps}
        expected = {directory / f"tick-{tick:04d}-ant-{focal:02d}.residual.npz" for tick in ticks}
        if set(directory.glob(f"*-ant-{focal:02d}.residual.npz")) != expected:
            raise ValueError("快照不完整")
        for path in expected:
            with np.load(path, allow_pickle=False) as data:
                if (str(data["adapter_kind"]) != "direction-trust" or int(data["writes"]) != 0
                        or np.any(data["stable"]) or not np.array_equal(data["fast"], initial.flatten())):
                    raise ValueError("主轨迹参数并非固定初始值")
            snapshots += 1
    rows = [CandidateRecord.model_validate_json(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
    if (len(rows) != world.pairs or len(rows) > plan.pairs_per_world
            or len({(r.tick, r.focal) for r in rows}) != len(rows)):
        raise ValueError("采样计数或身份重复")
    for row in rows:
        if (row.seed != world.seed or row.tick % 64 or row.proposal.steps != plan.adaptation.window
                or not 0 <= row.focal < plan.environment.ants or row.condition != "benign"
                or row.restoration is None or row.diagnostics is None):
            raise ValueError("记录身份或反馈范围错误")
        if len(row.observation) != 78 or len(row.hidden) != 8 or len(row.proposal.delta) != 45:
            raise ValueError("输入维度错误")
        if not np.isfinite([*row.observation, *row.hidden, *row.proposal.delta]).all():
            raise ValueError("输入非有限")
        if row.result.skip != row.restoration.skip or not row.restoration.changed or np.any(row.restoration.accepted_weights):
            raise ValueError("恢复参照错误")
        for branch in (row.result.accept, row.result.skip, row.restoration.accept):
            if branch.colony_deaths or branch.focal_injury or not 0 <= branch.steps <= plan.branch_horizon:
                raise ValueError("无危险条件或评价范围错误")
    return rows, snapshots


def run(config: Config) -> Result:
    files = verify_files(config)
    executions = [Execution.model_validate_json((config.input_directory / f"window-{w}/execution.json").read_text()) for w in (16, 64)]
    short, long = executions
    expected = short.plan.model_dump()
    expected["adaptation"]["window"] = 64
    if expected != long.plan.model_dump() or short.sources != long.sources:
        raise ValueError("两组除了反馈窗口以外还有其他差异")
    if (short.plan.sample_interval != 64 or not short.plan.include_zero_candidates
            or short.plan.initial_residual_norm != .75 or short.plan.initial_residual_pattern != "coherent-fourth"):
        raise ValueError("共同状态协议不成立")
    for execution, window in zip(executions, (16, 64), strict=True):
        if execution.smoke or execution.plan.adaptation.window != window:
            raise ValueError("短流程或错误窗口")
        if hashlib.sha256(Path(execution.plan_path).read_bytes()).hexdigest() != execution.plan_sha256:
            raise ValueError("协议散列改变")
        for source in execution.sources:
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
                raise ValueError("来源模型改变")
    world_sets = [[WorldRecord.model_validate_json(line) for line in
                   (config.input_directory / f"window-{w}/worlds.jsonl").read_text().splitlines()] for w in (16, 64)]
    for worlds in world_sets:
        if [w.seed for w in worlds] != list(short.plan.seeds):
            raise ValueError("世界不完整或顺序不一致")
    results: list[WorldResult] = []
    snapshots = 0
    for a, b in zip(*world_sets, strict=True):
        if (a.seed, a.steps, a.deliveries, a.deaths, a.pairs) != (b.seed, b.steps, b.deliveries, b.deaths, b.pairs):
            raise ValueError("世界轨迹或配对计数不一致")
        dirs = [config.input_directory / f"window-{w}/benign-{a.seed}" for w in (16, 64)]
        with gzip.open(dirs[0] / "parent.jsonl.gz", "rt") as left, gzip.open(dirs[1] / "parent.jsonl.gz", "rt") as right:
            if left.read() != right.read():
                raise ValueError("两组完整主轨迹不一致")
        rows = []
        for directory, execution, world in zip(dirs, executions, (a, b), strict=True):
            records, count = audit(directory, execution, world)
            snapshots += count
            rows.append(records)
        for first, second in zip(*rows, strict=True):
            if ((first.tick, first.focal, first.observation, first.hidden, first.current_rotation, first.restoration, first.result.skip)
                    != (second.tick, second.focal, second.observation, second.hidden, second.current_rotation, second.restoration, second.result.skip)):
                raise ValueError("候选不是同一状态或共同随机延续")
        results.append(WorldResult(seed=a.seed, pairs=a.pairs, windows=[summarize(w, rs) for w, rs in zip((16, 64), rows, strict=True)],
            long_minus_short_reward=describe([v.result.accept.focal_reward - u.result.accept.focal_reward for u, v in zip(*rows, strict=True)]),
            long_minus_short_focal_deliveries=describe([float(v.result.accept.focal_deliveries - u.result.accept.focal_deliveries) for u, v in zip(*rows, strict=True)], 0.),
            long_minus_short_colony_deliveries=describe([float(v.result.accept.colony_deliveries - u.result.accept.colony_deliveries) for u, v in zip(*rows, strict=True)], 0.)))
    select = all(r.pairs and r.long_minus_short_reward.mean > 0. for r in results)
    select = select and any(r.long_minus_short_focal_deliveries.mean is not None and r.long_minus_short_focal_deliveries.mean > 0. for r in results)
    result = Result(files_verified=files, snapshots_verified=snapshots, worlds=results, select_long_window=select)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(Config.model_validate_json(args.config.read_text())).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
