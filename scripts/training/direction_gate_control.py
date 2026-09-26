"""固定接受次数、随机接受时刻，排查学习判断是否只受益于少写参数。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel
import torch

from direction_adaptation import FOUNDATION, ROOT
from direction_gate import Config, Frame, HORIZON, Result, SCENARIOS, evaluate
from mathhackson.training.direction.gate import UpdateGate

REFERENCE = ROOT / "logs/direction-gate/20260926T060954-2"


def random_schedule(horizon: int, count: int, seed: tuple[int, ...]) -> np.ndarray:
    if horizon < 1 or not 0 <= count <= horizon:
        raise ValueError("接受预算必须处于轨迹长度以内")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    schedule = np.zeros(horizon, dtype=np.bool_)
    schedule[rng.choice(horizon, count, replace=False)] = True
    return schedule


class Artifact(BaseModel):
    path: str
    sha256: str


class ControlConfig(BaseModel):
    source_commit: str
    reference: str
    artifacts: list[Artifact]
    random_seeds: tuple[int, ...] = (6101, 6102, 6103, 6104, 6105)
    purpose: str = "开发数据的事后等次数时刻对照，不是可部署策略或正式泛化验证"


class Pair(BaseModel):
    reference_index: int
    random_seed: int
    reference: Result
    control: Result
    acceptance_count_matches: bool
    actual_write_count_matches: bool
    learned_minus_random_error: float
    learned_minus_random_progress: float


def identify(path: Path) -> Artifact:
    return Artifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_reference(result: Result, frames: list[Frame]) -> None:
    if len(frames) != HORIZON or any(frame.tick != tick for tick, frame in enumerate(frames)):
        raise ValueError("旧轨迹缺步或顺序不正确")
    if result.accepts != sum(frame.accept for frame in frames):
        raise ValueError("旧接受预算与原始帧记录不符")
    if result.updates != sum(frame.offset_before != frame.offset_after for frame in frames):
        raise ValueError("旧实际写入次数与帧记录不符")
    if not result.motor_unchanged or frames[0].offset_before != 0.:
        raise ValueError("旧轨迹未满足冻结或初始化约束")
    if any(not f.accept and f.offset_after != f.offset_before for f in frames):
        raise ValueError("旧轨迹拒绝时修改了行为参数")
    error = float(np.mean([f.error_degrees for f in frames[64:]]))
    progress = float(np.mean([f.progress_fraction for f in frames[64:]]))
    if error != result.mean_error_degrees or progress != result.progress_fraction:
        raise ValueError("旧结果与原始轨迹重新计算不一致")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    reference_config = Config.model_validate_json((REFERENCE / "config.json").read_text())
    references = [result for line in (REFERENCE / "results.jsonl").read_text().splitlines()
                  if (result := Result.model_validate_json(line)).mode == "learned"]
    expected = {(e.model_dump_json(), seed) for e in reference_config.evaluation
                for seed in reference_config.gate_seeds}
    actual = {(r.episode.model_dump_json(), r.gate_seed) for r in references}
    if actual != expected or len(references) != len(expected):
        raise ValueError("旧结果不是完整且唯一的学习组集合")
    paths = [REFERENCE / "config.json", REFERENCE / "results.jsonl",
             ROOT / "logs/external-models/mulo/model.safetensors"]
    paths.extend(REFERENCE / f"gate-{seed}/update-000600.npz" for seed in reference_config.gate_seeds)
    paths.extend(FOUNDATION / f"seed-{seed}/update-001200.npz" for seed in (41, 42, 43))
    frame_paths: list[Path] = []
    for reference in references:
        episode_index = reference_config.evaluation.index(reference.episode)
        path = REFERENCE / f"episode-{episode_index:02d}-learned-{reference.gate_seed}/frames.jsonl"
        validate_reference(reference, [Frame.model_validate_json(line) for line in path.read_text().splitlines()])
        frame_paths.append(path)
    paths.extend(frame_paths)
    config = ControlConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                           reference=str(REFERENCE), artifacts=[identify(path) for path in paths])
    (directory / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    # 先验证复用执行器未改变学习组行为；不使用新表现挑选参考模型。
    first = references[0]
    reproduction = directory / "reference-reproduction"
    reproduction.mkdir()
    gate = UpdateGate.load(REFERENCE / f"gate-{first.gate_seed}/update-000600.npz")
    evaluate(first.episode, "learned", first.gate_seed, gate, reproduction)
    if (reproduction / "frames.jsonl").read_bytes() != frame_paths[0].read_bytes():
        raise AssertionError("原学习轨迹未逐帧精确复现，停止新增对照")
    for index, reference in enumerate(references):
        for random_seed in config.random_seeds:
            assert reference.gate_seed is not None
            episode = reference.episode
            seed = (random_seed, episode.motor_seed, episode.scene_seed,
                    reference.gate_seed, SCENARIOS.index(episode.scenario))
            schedule = random_schedule(HORIZON, reference.accepts, seed)
            path = directory / f"reference-{index:03d}-random-{random_seed}"
            path.mkdir()
            with (path / "schedule.npy").open("xb") as stream:
                np.save(stream, schedule, allow_pickle=False)
            control = evaluate(episode, "random", reference.gate_seed, None, path, schedule=schedule)
            control.error_difference = control.mean_error_degrees - (reference.mean_error_degrees - reference.error_difference)
            control.progress_difference = control.progress_fraction - (reference.progress_fraction - reference.progress_difference)
            pair = Pair(reference_index=index, random_seed=random_seed, reference=reference, control=control,
                        acceptance_count_matches=reference.accepts == control.accepts,
                        actual_write_count_matches=reference.updates == control.updates,
                        learned_minus_random_error=reference.mean_error_degrees - control.mean_error_degrees,
                        learned_minus_random_progress=reference.progress_fraction - control.progress_fraction)
            with (directory / "pairs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(pair.model_dump_json() + "\n")
            if not pair.acceptance_count_matches:
                raise AssertionError("随机对照未实现规定接受预算")
        print(f"等次数对照 {index + 1}/{len(references)}", flush=True)
    if [identify(path) for path in paths] != config.artifacts:
        raise AssertionError("运行期间既有输入产物变化")


if __name__ == "__main__":
    main()
