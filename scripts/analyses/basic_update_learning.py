"""固定基础课程的独立接受训练与留出后果描述。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.foraging.update_curriculum import (
    CurriculumExecution, CurriculumWorld, FitRecord, fit_individual, read_partition, world_key,
)
from mathhackson.training.foraging.update_decision import UpdateDecision


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_directory: Path
    manifest: Path
    output_directory: Path


class WorldScore(BaseModel):
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


class Result(BaseModel):
    verified_files: int
    verified_snapshots: int
    fits: list[FitRecord]
    worlds: list[WorldScore]
    seeds: list[SeedScore]
    development_continue: bool


def verify_inputs(config: Config, execution: CurriculumExecution) -> tuple[int, int]:
    if execution.smoke:
        raise ValueError("短流程不能作为科研分析数据")
    paths: set[Path] = set()
    for line in config.manifest.read_text().splitlines():
        expected, location = line.split("  ", 1)
        path = Path(location)
        if path in paths or not path.resolve().is_relative_to(config.input_directory.resolve()):
            raise ValueError("清单重复或超出当前数据集")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"文件散列不一致：{path}")
        paths.add(path)
    if paths != {path for path in config.input_directory.rglob("*") if path.is_file()}:
        raise ValueError("清单必须完整覆盖原始目录")
    for item in execution.sources:
        if hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256:
            raise ValueError("基础模型来源散列改变")
    plan = execution.plan
    worlds = [CurriculumWorld.model_validate_json(line) for line in (config.input_directory / "worlds.jsonl").read_text().splitlines()]
    expected_keys = {world_key(i, seed) for i, _ in enumerate(plan.initial_norms) for seed in plan.train_seeds + plan.held_seeds}
    if {world.key for world in worlds} != expected_keys or len(worlds) != len(expected_keys):
        raise ValueError("世界缺失或重复")
    snapshots = 0
    for world in worlds:
        index = plan.initial_norms.index(world.initial_norm)
        if (world.key != world_key(index, world.result.seed) or world.result.deaths
                or world.partition != ("train" if world.result.seed in plan.train_seeds else "held")):
            raise ValueError("世界身份或无危险条件不成立")
        directory = config.input_directory / world.key
        if len((directory / "pairs.jsonl").read_text().splitlines()) != world.result.pairs:
            raise ValueError("世界候选数不一致")
        for focal in range(plan.probe.environment.ants):
            initial = np.zeros(45, dtype=np.float32)
            if world.initial_norm:
                with np.load(directory / f"tick-0000-ant-{focal:02d}.residual.npz", allow_pickle=False) as data:
                    initial = data["fast"].copy()
                if not np.isclose(np.linalg.norm(initial), world.initial_norm, atol=1e-6):
                    raise ValueError("初始残差范数错误")
            ticks = list(range(64, world.result.steps + 1, 64)) + [world.result.steps]
            if world.initial_norm:
                ticks.append(0)
            expected = {directory / f"tick-{tick:04d}-ant-{focal:02d}.residual.npz" for tick in ticks}
            if set(directory.glob(f"*-ant-{focal:02d}.residual.npz")) != expected:
                raise ValueError("参数快照缺失或额外")
            for path in expected:
                with np.load(path, allow_pickle=False) as data:
                    if (str(data["adapter_kind"]) != "direction-trust" or int(data["writes"]) != 0
                            or not np.array_equal(data["fast"], initial) or np.any(data["stable"])):
                        raise ValueError("主轨迹残差发生非预期写入")
                snapshots += 1
    return len(paths), snapshots


def run(config: Config) -> Result:
    execution = CurriculumExecution.model_validate_json((config.input_directory / "execution.json").read_text())
    count, snapshots = verify_inputs(config, execution)
    plan = execution.plan
    training = read_partition(config.input_directory, plan, "train")
    config.output_directory.mkdir(parents=True, exist_ok=False)
    fits = [fit_individual(training, focal, plan, config.output_directory / f"ant-{focal:02d}")
            for focal in range(plan.probe.environment.ants)]
    # 只有所有训练结束后才读取留出标签，最终轮次固定，不能据此选择模型。
    held = read_partition(config.input_directory, plan, "held")
    models = [UpdateDecision.load(config.output_directory / f"ant-{focal:02d}" / f"step-{plan.training_steps:04d}.npz").eval()
              for focal in range(plan.probe.environment.ants)]
    worlds: list[WorldScore] = []
    for index, norm in enumerate(plan.initial_norms):
        for seed in plan.held_seeds:
            key = world_key(index, seed)
            rows = [row for row in held if row.key == key]
            with torch.inference_mode():
                predictions = np.asarray([float(models[row.focal](torch.from_numpy(row.features))) for row in rows])
            labels = np.asarray([row.benefit for row in rows])
            accepted = predictions > 0.
            focal_deliveries = np.asarray([row.record.result.accept.focal_deliveries - row.record.result.skip.focal_deliveries for row in rows])
            colony_deliveries = np.asarray([row.record.result.accept.colony_deliveries - row.record.result.skip.colony_deliveries for row in rows])
            if not np.isfinite(predictions).all():
                raise ValueError("留出预测非有限")
            worlds.append(WorldScore(key=key, seed=seed, initial_norm=norm, count=len(rows), accepted=int(accepted.sum()),
                                     positive=int((labels > 1e-6).sum()), negative=int((labels < -1e-6).sum()), tied=int((np.abs(labels) <= 1e-6).sum()),
                                     selected_reward=float((labels * accepted).mean()) if rows else None,
                                     always_reward=float(labels.mean()) if rows else None,
                                     prediction_mse=float(np.square(labels - predictions).mean()) if rows else None,
                                     selected_focal_deliveries=float((focal_deliveries * accepted).mean()) if rows else None,
                                     selected_colony_deliveries=float((colony_deliveries * accepted).mean()) if rows else None))
    seeds: list[SeedScore] = []
    for seed in plan.held_seeds:
        rows = [world for world in worlds if world.seed == seed]
        complete = all(world.count for world in rows)
        seeds.append(SeedScore(seed=seed, complete=bool(complete),
                               selected_reward=float(np.mean([world.selected_reward for world in rows])) if complete else None,
                               always_reward=float(np.mean([world.always_reward for world in rows])) if complete else None))
    proceed = all(row.complete and row.selected_reward > max(0., row.always_reward) for row in seeds)
    proceed = proceed and any(world.selected_focal_deliveries is not None and world.selected_focal_deliveries > 0. for world in worlds)
    result = Result(verified_files=count, verified_snapshots=snapshots, fits=fits, worlds=worlds, seeds=seeds, development_continue=proceed)
    with (config.output_directory / "summary.json").open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(Config.model_validate_json(args.config.read_text()))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
