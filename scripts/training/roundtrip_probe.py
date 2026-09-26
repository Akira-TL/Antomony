"""固定场景下的往返训练开发探针；结果不是正式研究比较。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mathhackson.training.recurrent import WriteMode
from mathhackson.training.roundtrip_environment import RoundTripEnvironment
from mathhackson.training.roundtrip_policy import ROUNDTRIP_MODEL_VERSION, RoundTripPolicy
from mathhackson.training.roundtrip_session import RoundTripSession
from mathhackson.training.roundtrip_session import ROUNDTRIP_GROUP_IDS
from mathhackson.training.scent_curriculum import (MemoryMetrics, ScentMetrics, TeacherMetrics,
                                                   evaluate_scent_memory,
                                                   evaluate_teacher_scent,
                                                   pretrain_scent_memory,
                                                   pretrain_scent_reader,
                                                   pretrain_teacher_scent)
from mathhackson.training.schemas import RoundTripCommand

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = ROOT / "checkpoints" / "recurrent" / "foundation-episode-002570.npz"


class Evaluation(BaseModel):
    name: str
    first_pickups: int
    first_deliveries: int
    two_deliveries: int
    total_deliveries: int
    seeds: int
    return_steps: int
    return_scent_steps: int
    early_return_steps: int
    early_return_scent_steps: int
    second_departure_steps: int
    food_scent_steps: int
    runtime_writes: int
    mean_write_probability: float


class ProbeReport(BaseModel):
    session: str
    source_checkpoint: str | None = None
    scent_steps: int
    scent_turn_error: float | None = None
    scent_conflict_accuracy: float | None = None
    memory_pretrain_steps: int
    memory_before: MemoryMetrics | None = None
    memory_after: MemoryMetrics | None = None
    teacher_steps: int
    teacher_before: TeacherMetrics | None = None
    teacher_after: TeacherMetrics | None = None
    memory_episodes: int
    adaptive_episodes: int
    evaluations: list[Evaluation]


def evaluate(model: RoundTripPolicy, name: str, seeds: int, write_mode: WriteMode,
             scent_enabled: bool = True) -> Evaluation:
    previous_phase, previous_mode = model.phase, model.write_mode
    model.phase = "autonomous"
    model.write_mode = write_mode
    delivered: list[int] = []
    pickups: list[int] = []
    writes = 0
    probabilities = 0.
    steps = 0
    return_steps = return_scent_steps = second_departure_steps = food_scent_steps = 0
    early_return_steps = early_return_scent_steps = 0
    try:
        for seed in range(seeds):
            environment = RoundTripEnvironment(seed)
            model.reset_state()
            while not environment.done:
                observation = environment.observation()
                if environment.carrying:
                    return_steps += 1
                    scented = int(np.linalg.norm(observation[9:11]) >= .005)
                    return_scent_steps += scented
                    if environment.leg_steps < 20:
                        early_return_steps += 1
                        early_return_scent_steps += scented
                elif environment.delivered:
                    second_departure_steps += 1
                    food_scent_steps += int(np.linalg.norm(observation[12:14]) >= .005)
                action = model.decide(observation)
                environment.step(action.move, action.turn,
                                 action.release_home if scent_enabled else False,
                                 action.release_food if scent_enabled else False)
                write = model.observe_result(environment.observation(), terminal=environment.done)
                writes += int(write.wrote)
                probabilities += write.probability
                steps += 1
            delivered.append(environment.delivered)
            pickups.append(environment.pickups)
    finally:
        model.phase, model.write_mode = previous_phase, previous_mode
        model.reset_state()
    return Evaluation(name=name, first_pickups=sum(value >= 1 for value in pickups),
                      first_deliveries=sum(value >= 1 for value in delivered),
                      two_deliveries=sum(value >= 2 for value in delivered),
                      total_deliveries=sum(delivered), seeds=seeds,
                      return_steps=return_steps, return_scent_steps=return_scent_steps,
                      early_return_steps=early_return_steps,
                      early_return_scent_steps=early_return_scent_steps,
                      second_departure_steps=second_departure_steps,
                      food_scent_steps=food_scent_steps,
                      runtime_writes=writes, mean_write_probability=probabilities / max(steps, 1))


def train(session: RoundTripSession, episodes: int) -> None:
    last = session.episode + episodes
    while session.episode < last:
        session.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=int, default=50)
    parser.add_argument("--adaptive", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--scent-steps", type=int, default=0)
    parser.add_argument("--memory-pretrain-steps", type=int, default=0)
    parser.add_argument("--teacher-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    if min(arguments.memory, arguments.adaptive, arguments.scent_steps,
           arguments.memory_pretrain_steps, arguments.teacher_steps) < 0 or arguments.seeds < 1:
        parser.error("训练次数不能为负，验收场景数至少为 1")
    session = RoundTripSession(ROOT / "logs" / "roundtrip-probe", FOUNDATION, seed=91)
    if arguments.checkpoint is not None:
        session.model.load_roundtrip_checkpoint(arguments.checkpoint)
        evaluations = [evaluate(session.model, "载入检查点，无写入", arguments.seeds, "off")]
    else:
        evaluations = [evaluate(session.model, "初始，无写入", arguments.seeds, "off")]
        evaluations.append(evaluate(session.model, "初始，禁用释放", arguments.seeds, "off", False))
    scent_metrics = None
    if arguments.scent_steps:
        def save_scent_checkpoint(step: int, metrics: ScentMetrics) -> None:
            np.savez_compressed(
                session.directory / f"scent-step-{step:06d}.npz",
                model_version=ROUNDTRIP_MODEL_VERSION, stage="scent", step=step,
                curriculum_seed=20260926, batch_size=256,
                turn_error=metrics.turn_error,
                conflict_accuracy=metrics.conflict_accuracy,
                **{identity: parameter.detach().numpy() for identity, parameter in
                   zip(ROUNDTRIP_GROUP_IDS, session.model.parameters, strict=True)},
            )

        scent_metrics = pretrain_scent_reader(session.model, steps=arguments.scent_steps,
                                              checkpoint=save_scent_checkpoint)
        evaluations.append(evaluate(session.model, "局部气味预训练，无写入", arguments.seeds, "off"))
    memory_before = memory_after = None
    if arguments.memory_pretrain_steps:
        memory_before = evaluate_scent_memory(session.model)

        def save_memory_checkpoint(step: int, metrics: MemoryMetrics) -> None:
            np.savez_compressed(
                session.directory / f"memory-step-{step:06d}.npz",
                model_version=ROUNDTRIP_MODEL_VERSION, stage="memory", step=step,
                curriculum_seed=20260929, batch_size=64,
                query_turn_error=metrics.full.turn_error,
                query_direction_accuracy=metrics.full.direction_accuracy,
                no_memory_direction_accuracy=metrics.without_memory.direction_accuracy,
                short_only_direction_accuracy=metrics.short_only.direction_accuracy,
                long_only_direction_accuracy=metrics.long_only.direction_accuracy,
                **{identity: parameter.detach().numpy() for identity, parameter in
                   zip(ROUNDTRIP_GROUP_IDS, session.model.parameters, strict=True)},
            )

        memory_after = pretrain_scent_memory(session.model, steps=arguments.memory_pretrain_steps,
                                             checkpoint=save_memory_checkpoint)
        evaluations.append(evaluate(session.model, "间断气味预训练，无写入", arguments.seeds, "off"))
    teacher_before = teacher_after = None
    if arguments.teacher_steps:
        teacher_before = evaluate_teacher_scent(session.model)

        def save_teacher_checkpoint(step: int, metrics: TeacherMetrics) -> None:
            np.savez_compressed(
                session.directory / f"teacher-step-{step:06d}.npz",
                model_version=ROUNDTRIP_MODEL_VERSION, stage="teacher", step=step,
                train_seed=1000, train_seeds=64, validation_seed=2000, validation_seeds=16,
                return_turn_error=metrics.return_error,
                cue_missing_turn_error=metrics.cue_missing_error,
                **{identity: parameter.detach().numpy() for identity, parameter in
                   zip(ROUNDTRIP_GROUP_IDS, session.model.parameters, strict=True)},
            )

        teacher_after = pretrain_teacher_scent(session.model, steps=arguments.teacher_steps,
                                                checkpoint=save_teacher_checkpoint)
        evaluations.append(evaluate(session.model, "真实轨迹基础训练，无写入", arguments.seeds, "off"))
    train(session, arguments.memory)
    if arguments.memory:
        evaluations.append(evaluate(session.model, "基础循迹，无写入", arguments.seeds, "off"))
    if arguments.adaptive:
        session.command(RoundTripCommand(action="phase", phase="adaptive"))
        train(session, arguments.adaptive)
        for mode, name in (("off", "条件训练后，关闭写入"),
                           ("learned", "条件训练后，模型判断"),
                           ("always", "条件训练后，始终写入")):
            evaluations.append(evaluate(session.model, name, arguments.seeds, mode))
    report = ProbeReport(session=session.id,
                         source_checkpoint=str(arguments.checkpoint) if arguments.checkpoint else None,
                         scent_steps=arguments.scent_steps,
                         scent_turn_error=scent_metrics.turn_error if scent_metrics else None,
                         scent_conflict_accuracy=scent_metrics.conflict_accuracy if scent_metrics else None,
                         memory_pretrain_steps=arguments.memory_pretrain_steps,
                         memory_before=memory_before, memory_after=memory_after,
                         teacher_steps=arguments.teacher_steps,
                         teacher_before=teacher_before, teacher_after=teacher_after,
                         memory_episodes=arguments.memory,
                         adaptive_episodes=arguments.adaptive, evaluations=evaluations)
    path = session.directory / "probe.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    print(f"记录：{path}")


if __name__ == "__main__":
    main()
