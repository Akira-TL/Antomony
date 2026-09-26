"""只使用临时合成课程记录，验证审计拒绝篡改；不构成训练效果证据。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from mathhackson.training.direction.policy import DirectionMotor
from mathhackson.training.foraging.feedback_cues import CONDITIONS, CueConfig, cue_episode
from mathhackson.training.foraging.plastic_direction import PlasticDirection
from mathhackson.training.foraging.plastic_course.rollout import CourseTrace, Mode, matched_schedule, rollout
from mathhackson.training.foraging.plastic_course.run import EvaluationRow, Initialization, Protocol, Source, save_trace
from mathhackson.training.foraging.reward import direction_distribution
from scripts.analyses.feedback_plasticity import audited_rows


MODES: tuple[Mode, ...] = ("learned", "off", "always", "matched")


@dataclass(frozen=True)
class AuditFixture:
    directory: Path
    plan: Protocol

    def trace_path(self, mode: Mode = "learned") -> Path:
        initialization = self.plan.initializations[0]
        return self.directory / f"seed-{initialization.seed}" / f"reference-{initialization.evaluation_seed}-{mode}.npz"


@pytest.fixture
def records(tmp_path: Path) -> AuditFixture:
    # 来源仅满足协议结构；audited_rows 不读取这些路径，不加载正式模型。
    source = Source(path="unused-synthetic-model.npz", sha256="0" * 64)
    plan = Protocol(motor=source, updates=25, evaluation_batches=1,
        course=CueConfig(steps=8, batch=1), initializations=(
            Initialization(seed=11, foundation=source, train_seed=100, evaluation_seed=200),
            Initialization(seed=12, foundation=source, train_seed=300, evaluation_seed=400)))
    for initialization in plan.initializations:
        directory = tmp_path / f"seed-{initialization.seed}"
        directory.mkdir()
        model = PlasticDirection(DirectionMotor(initialization.seed), seed=initialization.seed,
            max_step=plan.max_step, max_fast=plan.max_fast)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(torch.tensor([.5, 10.]))
        model.eval().requires_grad_(False)
        rows: list[EvaluationRow] = []
        for condition in CONDITIONS:
            seed = initialization.evaluation_seed
            episode = cue_episode(seed, plan.course, condition=condition)
            base = torch.tensor([.6, .8]).repeat(plan.course.steps, plan.course.batch, 1)
            learned: CourseTrace | None = None
            for mode in MODES:
                schedule = matched_schedule(learned.accepted, seed + 20_000_000) if mode == "matched" and learned else None
                with torch.no_grad():
                    trace = rollout(model, episode, base, seed=seed + 10_000_000, mode=mode, schedule=schedule)
                if mode == "learned":
                    learned = trace
                save_trace(directory / f"{condition}-{seed}-{mode}.npz", episode, base, trace,
                    structure_update=plan.updates)
                rows.append(EvaluationRow(initialization=initialization.seed, condition=condition,
                    batch_seed=seed, mode=mode,
                    mean_after_feedback=trace.rewards[4:].mean(dim=0).tolist(),
                    mean_last_quarter=trace.rewards[-plan.course.steps // 4:].mean(dim=0).tolist(),
                    acceptances=trace.accepted.sum(dim=0).tolist(),
                    nonzero_writes=(trace.write_norm > 1e-9).sum(dim=0).tolist(),
                    total_variation_from_initial=(.5 * (trace.probabilities - direction_distribution(base).probs)
                        .abs().sum(dim=-1)).mean(dim=0).tolist()))
        (directory / "evaluations.jsonl").write_text(
            "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    return AuditFixture(tmp_path, plan)


def test_audit_accepts_complete_paired_synthetic_records(records: AuditFixture) -> None:
    rows = audited_rows(records.directory, records.plan)
    assert len(rows) == 2 * 3 * 1 * 4
    assert {(row.initialization, row.condition, row.mode) for row in rows} == {
        (initialization.seed, condition, mode)
        for initialization in records.plan.initializations for condition in CONDITIONS for mode in MODES}
    assert all(row.acceptances == ([0] if row.mode == "off" else [2]) for row in rows)
    assert all(row.nonzero_writes == ([0] if row.mode == "off" else [2]) for row in rows)


def test_audit_rejects_tampered_reported_acceptance_count(records: AuditFixture) -> None:
    path = records.trace_path().parent / "evaluations.jsonl"
    rows = [EvaluationRow.model_validate_json(line) for line in path.read_text().splitlines()]
    rows[0].acceptances[0] += 1
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="汇总与原始记录不一致"):
        audited_rows(records.directory, records.plan)


def test_audit_rejects_unmatched_actual_acceptance_count(records: AuditFixture) -> None:
    path = records.trace_path("matched")
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    arrays["accepted"][3, 0] = False
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="随机接受次数没有逐回合匹配"):
        audited_rows(records.directory, records.plan)


@pytest.mark.parametrize("claim_acceptance", [False, True])
def test_audit_rejects_write_outside_decision_frame(records: AuditFixture, claim_acceptance: bool) -> None:
    path = records.trace_path()
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    arrays["fast"][0, 0, 0, 0] = .01
    arrays["accepted"][0, 0] = claim_acceptance
    np.savez_compressed(path, **arrays)
    message = "接受记录或四帧边界无效" if claim_acceptance else "跳过、写入范数或参数约束不符"
    with pytest.raises(ValueError, match=message):
        audited_rows(records.directory, records.plan)


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf])
def test_audit_rejects_nonfinite_rewards(records: AuditFixture, reward: float) -> None:
    path = records.trace_path()
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    arrays["rewards"][0, 0] = reward
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="轨迹包含非有限值"):
        audited_rows(records.directory, records.plan)
