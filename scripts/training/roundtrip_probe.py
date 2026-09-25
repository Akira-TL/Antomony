"""固定场景下的往返训练开发探针；结果不是正式研究比较。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from mathhackson.training.recurrent import WriteMode
from mathhackson.training.roundtrip_environment import RoundTripEnvironment
from mathhackson.training.roundtrip_policy import RoundTripPolicy
from mathhackson.training.roundtrip_session import RoundTripSession
from mathhackson.training.schemas import RoundTripCommand

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = ROOT / "checkpoints" / "recurrent" / "foundation-episode-002570.npz"


class Evaluation(BaseModel):
    name: str
    first_deliveries: int
    two_deliveries: int
    total_deliveries: int
    seeds: int
    runtime_writes: int
    mean_write_probability: float


class ProbeReport(BaseModel):
    session: str
    memory_episodes: int
    adaptive_episodes: int
    evaluations: list[Evaluation]


def evaluate(model: RoundTripPolicy, name: str, seeds: int, write_mode: WriteMode) -> Evaluation:
    previous_phase, previous_mode = model.phase, model.write_mode
    model.phase = "autonomous"
    model.write_mode = write_mode
    delivered: list[int] = []
    writes = 0
    probabilities = 0.
    steps = 0
    try:
        for seed in range(seeds):
            environment = RoundTripEnvironment(seed)
            model.reset_state()
            while not environment.done:
                action = model.decide(environment.observation())
                environment.step(action.move, action.turn,
                                 action.release_home, action.release_food)
                write = model.observe_result(environment.observation(), terminal=environment.done)
                writes += int(write.wrote)
                probabilities += write.probability
                steps += 1
            delivered.append(environment.delivered)
    finally:
        model.phase, model.write_mode = previous_phase, previous_mode
        model.reset_state()
    return Evaluation(name=name, first_deliveries=sum(value >= 1 for value in delivered),
                      two_deliveries=sum(value >= 2 for value in delivered),
                      total_deliveries=sum(delivered), seeds=seeds,
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
    arguments = parser.parse_args()
    if min(arguments.memory, arguments.adaptive) < 0 or arguments.seeds < 1:
        parser.error("回合数不能为负，验收场景数至少为 1")
    session = RoundTripSession(ROOT / "logs" / "roundtrip-probe", FOUNDATION, seed=91)
    evaluations = [evaluate(session.model, "初始，无写入", arguments.seeds, "off")]
    train(session, arguments.memory)
    evaluations.append(evaluate(session.model, "基础循迹，无写入", arguments.seeds, "off"))
    if arguments.adaptive:
        session.command(RoundTripCommand(action="phase", phase="adaptive"))
        train(session, arguments.adaptive)
        for mode, name in (("off", "条件训练后，关闭写入"),
                           ("learned", "条件训练后，模型判断"),
                           ("always", "条件训练后，始终写入")):
            evaluations.append(evaluate(session.model, name, arguments.seeds, mode))
    report = ProbeReport(session=session.id, memory_episodes=arguments.memory,
                         adaptive_episodes=arguments.adaptive, evaluations=evaluations)
    path = session.directory / "probe.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    print(f"记录：{path}")


if __name__ == "__main__":
    main()
