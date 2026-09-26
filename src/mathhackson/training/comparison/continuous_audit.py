"""从固定输入重建连续部署；机械复现与科学效果判断分开。"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import tempfile

import numpy as np
from pydantic import BaseModel
import torch

from .auditing import verify_manifest
from .continuous import ARMS, ContinuousPlan, Execution, Frame, WorldResult, run_world
from .continuous_scores import Summary, summarize


class Window(BaseModel):
    seed: int
    condition: str
    arm: str
    start: int
    stop: int
    deliveries: int
    pickups: int
    deaths: int
    exhausted: int
    active_individual_steps: int
    reward: float
    writes: int


class AuditResult(BaseModel):
    files_verified: int
    frames_verified: int
    parameter_files_verified: int
    proposal_records_verified: int
    execution: Execution
    worlds: list[WorldResult]
    windows: list[Window]
    summary: Summary
    independent_simulator_validation: bool = False


def frames(directory: Path) -> list[Frame]:
    with gzip.open(directory / "trajectory.jsonl.gz", "rt") as stream:
        return [Frame.model_validate_json(line) for line in stream]


def compare_world(original: Path, reconstructed: Path, result: WorldResult, rebuilt: WorldResult) -> tuple[int, int]:
    if result.model_dump(exclude={"elapsed_seconds"}) != rebuilt.model_dump(exclude={"elapsed_seconds"}):
        raise ValueError("重建世界汇总与原始记录不符")
    if WorldResult.model_validate_json((original / "result.json").read_text()) != result:
        raise ValueError("世界文件与总表不符")
    left = {p.name for p in original.iterdir() if p.is_file()}
    right = {p.name for p in reconstructed.iterdir() if p.is_file()}
    if left != right:
        raise ValueError("世界产物缺失或多余")
    parameters = 0
    for name in sorted(left):
        a, b = original / name, reconstructed / name
        if name.endswith(".npz"):
            with np.load(a, allow_pickle=False) as saved, np.load(b, allow_pickle=False) as fresh:
                if set(saved.files) != set(fresh.files) or any(
                    not np.array_equal(saved[key], fresh[key]) for key in saved.files
                ):
                    raise ValueError(f"参数快照不能重建: {name}")
            parameters += 1
        elif name == "trajectory.jsonl.gz":
            with gzip.open(a, "rb") as saved, gzip.open(b, "rb") as fresh:
                if saved.read() != fresh.read():
                    raise ValueError("完整观察、动作、事件与写入轨迹不能重建")
        elif name == "updates.jsonl":
            if a.read_bytes() != b.read_bytes():
                raise ValueError("提案、接受或参数写入记录不能重建")
        elif name != "result.json":
            raise ValueError(f"未知世界产物: {name}")
    proposals = len((original / "updates.jsonl").read_text().splitlines())
    if proposals != result.decisions:
        raise ValueError("提案计数错误")
    return parameters, proposals


def window_scores(result: WorldResult, trace: list[Frame]) -> list[Window]:
    if len(trace) != result.steps or [f.tick for f in trace] != list(range(1, result.steps + 1)):
        raise ValueError("轨迹时间轴不连续")
    scores = []
    for start, stop in ((0, 128), (128, 384), (384, 768)):
        selected = trace[start:stop]
        before = trace[start - 1].ants if 0 < start <= len(trace) else []
        previous_deaths = sum(a.killed for a in before)
        previous_exhausted = sum(a.exhausted for a in before)
        previous_writes = sum(a.writes for a in before)
        end = selected[-1].ants if selected else before
        scores.append(Window(seed=result.seed, condition=result.condition, arm=result.arm, start=start, stop=stop,
            deliveries=sum(a.delivered for f in selected for a in f.ants),
            pickups=sum(a.picked_up for f in selected for a in f.ants),
            deaths=sum(a.killed for a in end) - previous_deaths if selected else 0,
            exhausted=sum(a.exhausted for a in end) - previous_exhausted if selected else 0,
            active_individual_steps=sum(a.active for f in selected for a in f.ants),
            reward=sum(a.reward for f in selected for a in f.ants),
            writes=sum(a.writes for a in end) - previous_writes if selected else 0))
    if (sum(w.deliveries for w in scores) != result.deliveries
            or sum(w.pickups for w in scores) != result.pickups
            or sum(w.deaths for w in scores) != result.deaths
            or sum(w.exhausted for w in scores) != result.exhausted
            or sum(w.writes for w in scores) != sum(result.writes)
            or sum(w.active_individual_steps for w in scores) != result.active_individual_steps
            or not np.isclose(sum(w.reward for w in scores), result.reward, rtol=1e-10, atol=1e-8)):
        raise ValueError("固定时间窗事件重建与汇总不符")
    return scores


def audit(directory: Path, manifest: Path, protocol: Path, work_directory: Path) -> AuditResult:
    files = verify_manifest(directory, manifest)
    execution = Execution.model_validate_json((directory / "execution.json").read_text())
    plan = execution.plan
    if (execution.smoke or not execution.completed_at or plan.environment.horizon != 768
            or plan != ContinuousPlan.model_validate_json(protocol.read_text())
            or execution.protocol_sha256 != hashlib.sha256(protocol.read_bytes()).hexdigest()):
        raise ValueError("正式执行或冻结协议身份不符")
    paths = {Path(plan.motor), *[plan.policy_path(i) for i in range(plan.environment.ants)],
             *[plan.gate_path(i) for i in range(plan.environment.ants)],
             *[plan.mlp_path(i) for i in range(plan.environment.ants)]}
    if len(execution.sources) != len(paths) or {Path(s.path) for s in execution.sources} != paths:
        raise ValueError("源模型身份缺失或重复")
    if any(hashlib.sha256(Path(s.path).read_bytes()).hexdigest() != s.sha256 for s in execution.sources):
        raise ValueError("输入模型改变")
    worlds = [WorldResult.model_validate_json(line) for line in (directory / "worlds.jsonl").read_text().splitlines()]
    expected = [(seed, c.name, arm) for seed in plan.seeds for c in plan.conditions for arm in ARMS]
    if [(w.seed, w.condition, w.arm) for w in worlds] != expected:
        raise ValueError("世界身份、顺序或数量不符")
    traces: dict[tuple[int, str], list[Frame]] = {}
    windows: list[Window] = []
    nframes = parameters = proposals = 0
    torch.set_num_threads(1)
    work_directory.mkdir(parents=True, exist_ok=True)
    for result in worlds:
        condition = next(c for c in plan.conditions if c.name == result.condition)
        original = directory / f"{result.condition}-{result.seed}-{result.arm}"
        with tempfile.TemporaryDirectory(dir=work_directory) as temporary:
            rebuilt_path = Path(temporary) / "world"
            rebuilt = run_world(plan, result.seed, condition, result.arm, rebuilt_path)
            p, u = compare_world(original, rebuilt_path, result, rebuilt)
        parameters += p
        proposals += u
        trace = frames(original)
        windows.extend(window_scores(result, trace))
        nframes += len(trace)
        key = (result.seed, result.condition)
        if result.arm == "learned":
            traces[key] = trace
        elif result.arm in ("skip", "always"):
            length = None if result.condition == "reference" else condition.disturbance.active_from
            if trace[:length] != traces[key][:length]:
                raise ValueError("三更新组参考轨迹或干预前轨迹不一致")
        print(f"verified {result.condition}-{result.seed}-{result.arm}", flush=True)
    return AuditResult(files_verified=files, frames_verified=nframes, parameter_files_verified=parameters,
        proposal_records_verified=proposals, execution=execution, worlds=worlds, windows=windows,
        summary=summarize(worlds, plan))
