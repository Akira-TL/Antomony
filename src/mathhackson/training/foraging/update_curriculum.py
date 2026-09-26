"""无危险基础更新课程，训练与留出按整巢种子隔离。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from .candidate_probe import CandidateRecord, ProbePlan, SourceArtifact, WorldRecord, collect
from .update_decision import UpdateDecision, decision_features, train_decision_step


class CurriculumPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    train_seeds: tuple[int, ...]
    held_seeds: tuple[int, ...]
    initial_norms: tuple[float, ...] = (0., .75)
    training_steps: int = Field(default=200, ge=1)
    checkpoint_every: int = Field(default=25, ge=1)
    learning_rate: float = Field(default=.01, gt=0.)
    weight_decay: float = Field(default=.01, ge=0.)
    probe: ProbePlan

    @model_validator(mode="after")
    def check_partition(self) -> CurriculumPlan:
        seeds = self.train_seeds + self.held_seeds
        if not self.train_seeds or not self.held_seeds or len(set(seeds)) != len(seeds):
            raise ValueError("训练与留出需要非空、无重复且不重叠的整巢种子")
        if set(seeds) != set(self.probe.seeds):
            raise ValueError("采样种子必须恰好覆盖训练与留出")
        if (not self.initial_norms or len(set(self.initial_norms)) != len(self.initial_norms)
                or any(not np.isfinite(x) or not 0. <= x < self.probe.adaptation.maximum_residual_norm for x in self.initial_norms)):
            raise ValueError("初始扰动范数必须唯一、有限且在安全范围内")
        if self.probe.conditions != ("benign",) or self.probe.disturbance.injury_per_step != 0.:
            raise ValueError("基础课程禁止危险机制训练")
        if self.probe.candidate_kind != "trust" or not self.probe.environment.ants <= 8:
            raise ValueError("基础课程使用概率约束候选和至多8份独立基础模型")
        return self


class CurriculumExecution(BaseModel):
    commit: str
    plan: CurriculumPlan
    plan_sha256: str
    sources: list[SourceArtifact]
    smoke: bool


class CurriculumWorld(BaseModel):
    key: str
    partition: Literal["train", "held"]
    initial_norm: float
    result: WorldRecord


def world_key(norm_index: int, seed: int) -> str:
    return f"initial-{norm_index}-seed-{seed}"


def collect_curriculum(plan: CurriculumPlan, output: Path, *, plan_sha256: str, smoke: bool = False) -> None:
    sources = [plan.probe.policy_path(i) for i in range(plan.probe.environment.ants)]
    sources.append(Path(plan.probe.motor_path))
    identities = [SourceArtifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    output.mkdir(parents=True, exist_ok=False)
    execution = CurriculumExecution(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                    plan=plan, plan_sha256=plan_sha256, sources=identities, smoke=smoke)
    (output / "execution.json").write_text(execution.model_dump_json(indent=2))
    with (output / "worlds.jsonl").open("x", encoding="utf-8") as stream:
        for index, norm in enumerate(plan.initial_norms):
            probe = plan.probe.model_copy(update={"initial_residual_norm": norm})
            for seed in plan.train_seeds + plan.held_seeds:
                key = world_key(index, seed)
                directory = output / key
                directory.mkdir()
                result = collect(probe, seed, "benign", directory)
                if result.deaths:
                    raise AssertionError("无危险基础课程不应出现死亡")
                record = CurriculumWorld(key=key, partition="train" if seed in plan.train_seeds else "held",
                                         initial_norm=norm, result=result)
                stream.write(record.model_dump_json() + "\n")
                stream.flush()
                print(record.model_dump_json(), flush=True)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in identities):
        raise AssertionError("采样改变了源模型")


@dataclass(frozen=True)
class LabeledDecision:
    key: str
    focal: int
    features: np.ndarray
    benefit: float
    record: CandidateRecord


def read_partition(directory: Path, plan: CurriculumPlan, partition: Literal["train", "held"]) -> list[LabeledDecision]:
    if partition not in ("train", "held"):
        raise ValueError("未知数据分区")
    seeds = plan.train_seeds if partition == "train" else plan.held_seeds
    rows: list[LabeledDecision] = []
    for index, _ in enumerate(plan.initial_norms):
        for seed in seeds:
            key = world_key(index, seed)
            seen: set[tuple[int, int]] = set()
            for line in (directory / key / "pairs.jsonl").read_text().splitlines():
                record = CandidateRecord.model_validate_json(line)
                identity = (record.tick, record.focal)
                if (record.seed != seed or record.condition != "benign" or record.diagnostics is None
                        or identity in seen or not 0 <= record.focal < plan.probe.environment.ants
                        or record.result.accept.colony_deaths or record.result.skip.colony_deaths
                        or not record.result.changed):
                    raise ValueError("课程记录身份、诊断或无危险约束不成立")
                seen.add(identity)
                benefit = record.result.accept.focal_reward - record.result.skip.focal_reward
                if not np.isfinite(benefit):
                    raise ValueError("配对标签非有限")
                features = decision_features(np.asarray(record.observation), np.asarray(record.hidden), record.proposal, record.diagnostics)
                rows.append(LabeledDecision(key, record.focal, features, benefit, record))
            if len(seen) > plan.probe.pairs_per_world:
                raise ValueError("采样超过冻结预算")
    return rows


class FitRecord(BaseModel):
    focal: int
    examples: int
    positive: int
    negative: int
    zero: int
    steps: int
    final_loss: float | None


def fit_individual(rows: list[LabeledDecision], focal: int, plan: CurriculumPlan, output: Path) -> FitRecord:
    selected = [row for row in rows if row.focal == focal]
    allowed = {world_key(index, seed) for index, _ in enumerate(plan.initial_norms) for seed in plan.train_seeds}
    if any(row.key not in allowed for row in selected):
        raise ValueError("留出世界不能参与接受决策训练")
    output.mkdir(parents=True, exist_ok=False)
    model = UpdateDecision()
    model.save(output / "step-0000.npz")
    final_loss = None
    labels = np.asarray([row.benefit for row in selected], dtype=np.float32)
    with (output / "loss.jsonl").open("x", encoding="utf-8") as stream:
        if selected:
            features = torch.from_numpy(np.stack([row.features for row in selected]))
            targets = torch.from_numpy(labels)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=plan.weight_decay)
            for step in range(1, plan.training_steps + 1):
                final_loss = train_decision_step(model, optimizer, features, targets)
                stream.write(TrainingLoss(step=step, loss=final_loss).model_dump_json() + "\n")
                if step % plan.checkpoint_every == 0 or step == plan.training_steps:
                    model.save(output / f"step-{step:04d}.npz")
    if not selected:
        model.save(output / f"step-{plan.training_steps:04d}.npz")
    return FitRecord(focal=focal, examples=len(selected), positive=int((labels > 1e-6).sum()),
                     negative=int((labels < -1e-6).sum()), zero=int((np.abs(labels) <= 1e-6).sum()),
                     steps=model.updates, final_loss=final_loss)


class TrainingLoss(BaseModel):
    step: int
    loss: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = CurriculumPlan.model_validate_json(args.plan.read_text())
    if args.smoke:
        probe = plan.probe.model_copy(update={"seeds": (9599, 9600), "pairs_per_world": 1, "branch_horizon": 3,
                                             "environment": plan.probe.environment.model_copy(update={"ants": 2, "horizon": 20})})
        plan = CurriculumPlan.model_validate(plan.model_dump() | {"train_seeds": (9599,), "held_seeds": (9600,), "probe": probe})
    torch.set_num_threads(1)
    collect_curriculum(plan, args.output, plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), smoke=args.smoke)


if __name__ == "__main__":
    main()
