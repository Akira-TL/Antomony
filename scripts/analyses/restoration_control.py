"""描述已知扰动恢复参照，不训练模型或执行部署控制。"""
from __future__ import annotations

import argparse
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
    count: int
    positive: int
    negative: int
    tied: int
    mean: float | None
    minimum: float | None
    maximum: float | None


def describe(values: list[float], tolerance: float = 1e-6) -> Difference:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("描述量必须有限")
    return Difference(count=len(values), positive=int((array > tolerance).sum()),
                      negative=int((array < -tolerance).sum()), tied=int((np.abs(array) <= tolerance).sum()),
                      mean=float(array.mean()) if values else None,
                      minimum=float(array.min()) if values else None,
                      maximum=float(array.max()) if values else None)


class OutcomeDifferences(BaseModel):
    focal_reward: Difference
    focal_deliveries: Difference
    colony_deliveries: Difference


class WorldResult(BaseModel):
    arm: str
    seed: int
    count: int
    absolute_rotation_degrees: Difference
    candidate: OutcomeDifferences
    restoration: OutcomeDifferences


class Result(BaseModel):
    verified_files: int
    verified_snapshots: int
    worlds: list[WorldResult]


def verify_manifest(config: Config) -> int:
    seen: set[Path] = set()
    for line in config.manifest.read_text().splitlines():
        expected, raw = line.split("  ", 1)
        path = Path(raw)
        if path in seen or not path.resolve().is_relative_to(config.input_directory.resolve()):
            raise ValueError("清单重复或超出当前数据集")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"文件散列不符：{path}")
        seen.add(path)
    if seen != {path for path in config.input_directory.rglob("*") if path.is_file()}:
        raise ValueError("清单未完整覆盖数据集")
    return len(seen)


def read_world(directory: Path, execution: Execution, world: WorldRecord) -> tuple[list[CandidateRecord], int]:
    plan = execution.plan
    snapshots = 0
    initial: list[np.ndarray] = []
    for focal in range(plan.environment.ants):
        fast = np.zeros(45, dtype=np.float32)
        if plan.initial_residual_norm:
            random = np.random.default_rng(world.seed * 8 + focal)
            if plan.initial_residual_pattern == "isotropic":
                value = random.normal(size=45)
            else:
                value = np.zeros((9, 5))
                value[:, 0] = random.choice((-1., 1.))
                value = value.flatten()
            fast = (value * (plan.initial_residual_norm / np.linalg.norm(value))).astype(np.float32)
        initial.append(fast)
        ticks = set(range(64, world.steps + 1, 64)) | {world.steps}
        if plan.initial_residual_norm:
            ticks.add(0)
        expected = {directory / f"tick-{tick:04d}-ant-{focal:02d}.residual.npz" for tick in ticks}
        if set(directory.glob(f"*-ant-{focal:02d}.residual.npz")) != expected:
            raise ValueError("主轨迹参数快照缺失或多余")
        for path in expected:
            with np.load(path, allow_pickle=False) as data:
                if (str(data["adapter_kind"]) != "direction-trust" or int(data["writes"]) != 0
                        or not np.array_equal(data["fast"], fast) or np.any(data["stable"])):
                    raise ValueError("主轨迹残差未保持初始值")
            snapshots += 1
    rows = [CandidateRecord.model_validate_json(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
    identities = {(row.tick, row.focal) for row in rows}
    if len(rows) != world.pairs or len(identities) != len(rows) or len(rows) > plan.pairs_per_world:
        raise ValueError("候选计数或唯一性错误")
    for row in rows:
        if (row.seed != world.seed or row.condition != "benign" or row.restoration is None
                or row.diagnostics is None or row.current_rotation is None or not 0 <= row.focal < plan.environment.ants):
            raise ValueError("候选缺少身份、恢复参照或角度")
        if len(row.observation) != 78 or len(row.hidden) != 8 or len(row.proposal.delta) != 45:
            raise ValueError("候选维度错误")
        if not np.isfinite([*row.observation, *row.hidden, *row.proposal.delta, row.current_rotation]).all():
            raise ValueError("候选包含非有限值")
        if row.result.skip != row.restoration.skip:
            raise ValueError("两次跳过延续不一致")
        if (not plan.initial_residual_norm and row.restoration.accept != row.restoration.skip
                or row.restoration.changed != bool(plan.initial_residual_norm)
                or len(row.restoration.accepted_weights) != 45 or np.any(row.restoration.accepted_weights)):
            raise ValueError("已知恢复不正确，或零残差负对照失效")
        for branch in (row.result.accept, row.result.skip, row.restoration.accept, row.restoration.skip):
            if not 0 <= branch.steps <= plan.branch_horizon or branch.colony_deaths or branch.focal_injury:
                raise ValueError("分支时限或无危险条件不符")
        novel = np.asarray(row.observation, dtype=np.float32)[:72].reshape(9, 8)[:, 3:].flatten()
        rotation = float(np.pi * np.tanh(initial[row.focal] @ novel))
        if not np.isclose(rotation, row.current_rotation, atol=1e-7):
            raise ValueError("记录方向角与初始参数不一致")
    return rows, snapshots


def outcomes(rows: list[CandidateRecord], *, restore: bool) -> OutcomeDifferences:
    pairs = [row.restoration if restore else row.result for row in rows]
    return OutcomeDifferences(
        focal_reward=describe([pair.accept.focal_reward - pair.skip.focal_reward for pair in pairs]),
        focal_deliveries=describe([float(pair.accept.focal_deliveries - pair.skip.focal_deliveries) for pair in pairs], 0.),
        colony_deliveries=describe([float(pair.accept.colony_deliveries - pair.skip.colony_deliveries) for pair in pairs], 0.),
    )


def run(config: Config) -> Result:
    verified = verify_manifest(config)
    results: list[WorldResult] = []
    snapshots = 0
    for arm in ("zero", "random", "coherent"):
        directory = config.input_directory / arm
        execution = Execution.model_validate_json((directory / "execution.json").read_text())
        if execution.smoke or not execution.plan.restoration_control:
            raise ValueError("短流程或非恢复协议不能进入分析")
        if (execution.plan.initial_residual_norm != (0. if arm == "zero" else .75)
                or execution.plan.initial_residual_pattern != ("coherent-fourth" if arm == "coherent" else "isotropic")):
            raise ValueError("条件目录与实际初始化不一致")
        if hashlib.sha256(Path(execution.plan_path).read_bytes()).hexdigest() != execution.plan_sha256:
            raise ValueError("采样协议散列改变")
        for source in execution.sources:
            if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256:
                raise ValueError("来源模型改变")
        worlds = [WorldRecord.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
        if {world.seed for world in worlds} != set(execution.plan.seeds) or len(worlds) != len(execution.plan.seeds):
            raise ValueError("世界缺失或重复")
        for world in worlds:
            if world.deaths or world.condition != "benign":
                raise ValueError("无危险条件不成立")
            rows, checked = read_world(directory / f"benign-{world.seed}", execution, world)
            snapshots += checked
            results.append(WorldResult(arm=arm, seed=world.seed, count=len(rows),
                                       absolute_rotation_degrees=describe([float(np.degrees(abs(row.current_rotation))) for row in rows]),
                                       candidate=outcomes(rows, restore=False), restoration=outcomes(rows, restore=True)))
    result = Result(verified_files=verified, verified_snapshots=snapshots, worlds=results)
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
