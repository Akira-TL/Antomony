from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from mathhackson.colony.geometry import unit

from .environment import SingleAntEnvironment
from .model import GROUPS, INPUTS, Decision, SelfModifyingPolicy
from .schemas import Command, EpisodeRecord, GroupState, State, TracePoint, WeightTrace


@dataclass(frozen=True)
class WeightSample:
    sequence: int
    tick: int
    episode: int
    phase: str
    source: str
    status: str
    weights: np.ndarray
    delta: np.ndarray


class TrainingSession:
    def __init__(self, directory: Path, seed: int = 20260925) -> None:
        self.id = uuid4().hex[:12]
        self.directory = directory / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = SelfModifyingPolicy(seed)
        self.env = SingleAntEnvironment(seed + 1)
        self.paused = True
        self.error = ""
        self.speed = 1
        self.tick = 0
        self.episode = 1
        self.baseline = 0.
        self.history: list[EpisodeRecord] = []
        self.trace: deque[WeightSample] = deque(maxlen=256)
        self._sample("initial", "初始参数", np.zeros((39, 16), dtype=np.float32))
        self._begin()

    def _sample(self, source: str, status: str, delta: np.ndarray) -> None:
        self.trace.append(WeightSample(self.trace[-1].sequence + 1 if self.trace else 0,
                                       self.tick, self.episode, self.model.phase, source, status,
                                       self.model.weights.detach().numpy().copy(), delta.copy()))

    def weight_trace(self, row: int, column: int) -> WeightTrace:
        if not 0 <= row < 39 or not 0 <= column < 16:
            raise ValueError("参数位置超出范围")
        points = [TracePoint(sequence=sample.sequence, tick=sample.tick, episode=sample.episode,
                             phase=sample.phase, source=sample.source, status=sample.status,
                             value=float(sample.weights[row, column]), delta=float(sample.delta[row, column]),
                             changed=int(np.count_nonzero(sample.delta)),
                             total_change=float(np.abs(sample.delta).sum())) for sample in self.trace]
        return WeightTrace(session=self.id, row=row, column=column, points=points)

    def motor_alignment(self) -> tuple[int, int]:
        probe = SelfModifyingPolicy(0)
        with torch.no_grad():
            probe.base.copy_(self.model.weights.detach())
        probe.phase = "autonomous"
        probe.manual_frozen = {group for group, _, _, _ in GROUPS}
        reached = [0, 0]
        for index in range(12):
            env = SingleAntEnvironment(index)
            env.max_turn = self.env.max_turn
            env.target = unit(-math.pi + (index + .5) * math.pi / 6.) * 4.
            env.distance = 4.
            while not env.done:
                decision = probe.decide(env.observation())
                env.step(decision.move, decision.turn)
            reached[index // 6] += int(env.reached)
        return reached[0], reached[1]

    def _begin(self) -> None:
        self.decisions: list[Decision] = []
        self.rewards: list[float] = []
        self.observations: list[np.ndarray] = []
        self.trajectory: list[np.ndarray] = []
        self.start = self.model.weights.detach().numpy().copy()
        self.start_self = self.model.self_updates
        self.start_outer = self.model.outer_updates

    def finish(self, reason: str, optimize: bool) -> None:
        if self.decisions:
            before_outer = self.model.weights.detach().numpy().copy()
            self.model.finish(self.decisions, self.rewards, self.baseline, optimize)
            outer_delta = self.model.outer_delta.numpy().copy()
            self._sample("outer", "回合末外部更新" if np.any(outer_delta) else "回合结束，参数未更新", outer_delta)
            filename = f"episode-{self.episode:06d}.npz"
            record = EpisodeRecord(episode=self.episode, phase=self.model.phase, lesson=self.env.lesson,
                                   steps=len(self.decisions), reward=sum(self.rewards), reached=self.env.reached,
                                   reason=reason, self_updates=self.model.self_updates - self.start_self,
                                   outer_updates=self.model.outer_updates - self.start_outer, checkpoint=filename)
            np.savez_compressed(self.directory / filename, start=self.start, before_outer=before_outer,
                                end=self.model.weights.detach().numpy(), weights=np.asarray(self.trajectory),
                                observations=np.asarray(self.observations), rewards=np.asarray(self.rewards),
                                actions=np.asarray([(d.move, d.turn) for d in self.decisions]),
                                gates=np.asarray([(d.requested_write, d.wrote) for d in self.decisions]),
                                frozen=(self.model.mask().numpy() == 0), max_turn=self.env.max_turn)
            (self.directory / filename.replace(".npz", ".json")).write_text(record.model_dump_json(indent=2), encoding="utf-8")
            self.history.append(record)
            self.history = self.history[-100:]
            self.episode += 1
            if optimize:
                # 仅用已结束回合建立基线，不窥看当前动作的未来结果。
                self.baseline = .9 * self.baseline + .1 * (sum(self.rewards) / len(self.rewards))
        self.env.reset()
        self._begin()

    def step(self) -> None:
        if self.env.done:
            self.env.reset()
        observation = self.env.observation()
        decision = self.model.decide(observation)
        reward = self.env.step(decision.move, decision.turn)
        self.decisions.append(decision)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.trajectory.append(self.model.weights.detach().numpy().copy())
        self.tick += 1
        self_delta = self.model.self_delta.numpy().copy()
        status = ("模型自写入" if decision.wrote else "基础阶段不自写入" if self.model.phase == "motor"
                  else "模型跳过写入" if not decision.requested_write else "冻结或变化为零")
        self._sample("self" if decision.wrote else "skip", status, self_delta)
        if self.env.done:
            self.finish("到达目标" if self.env.reached else "回合结束", True)

    def command(self, command: Command) -> None:
        action = command.action
        if action == "phase" and command.phase == "meta" and self.model.phase != "meta":
            left, right = self.motor_alignment()
            if left < 5 or right < 5:
                raise ValueError(f"基础动作尚未通过定向检查：两侧分别触达 {left}/6、{right}/6；各需至少 5/6。请先继续基础动作训练。")
        if action == "pause":
            self.paused = True
        elif action == "play":
            if self.error:
                raise ValueError(self.error)
            self.paused = False
        elif action == "step":
            self.paused = True
            if self.error:
                raise ValueError(self.error)
            self.step()
        elif action == "speed":
            self.speed = command.speed
        else:
            self.paused = True
            self.finish("配置改变，未执行外部更新", False)
            if action == "phase":
                self.model.phase = command.phase
            elif action == "lesson":
                self.env.lesson = command.lesson
            elif action == "turn":
                self.env.max_turn = command.max_turn
            elif action == "freeze":
                if command.frozen:
                    self.model.manual_frozen.add(command.group)
                else:
                    self.model.manual_frozen.discard(command.group)
            elif action == "freeze_all":
                self.model.manual_frozen = {group for group, _, _, _ in GROUPS} if command.frozen else set()
            self.env.reset()
            self._begin()

    def state(self) -> State:
        decision = self.decisions[-1] if self.decisions else None
        mask = self.model.mask()
        groups = [GroupState(id=group, label=label, start=start, end=end,
                             frozen=not bool(mask[start:end].any()), manual=group in self.model.manual_frozen,
                             reason="手动冻结" if group in self.model.manual_frozen else
                             f"基础阶段仅 {int(torch.count_nonzero(mask[start:end]))} 个连接可更新"
                             if self.model.phase == "motor" and group == "action" else
                             "基础阶段冻结" if self.model.phase == "motor" and group != "action" else "可更新")
                  for group, label, start, end in GROUPS]
        status = "尚未推理" if decision is None else "已写入" if decision.wrote else (
            "基础阶段禁用" if self.model.phase == "motor" else "模型跳过" if not decision.requested_write else "冻结，无变化")
        return State(session=self.id, paused=self.paused, error=self.error, phase=self.model.phase,
                     lesson=self.env.lesson, speed=self.speed, tick=self.tick, episode=self.episode,
                     steps=self.env.steps, horizon=self.env.horizon, x=float(self.env.position[0]), y=float(self.env.position[1]),
                     heading=self.env.heading, target_x=float(self.env.target[0]), target_y=float(self.env.target[1]),
                     distance=self.env.distance, max_turn=self.env.max_turn, move=self.env.move, turn=self.env.turn,
                     move_probability=decision.move_probability if decision else 0.,
                     write_probability=decision.write_probability if decision else 0., write_status=status,
                     reward=self.env.reward, total_reward=sum(self.rewards), self_updates=self.model.self_updates,
                     outer_updates=self.model.outer_updates, inputs=list(INPUTS),
                     observation=(self.observations[-1] if self.observations else self.env.observation()).tolist(),
                     groups=groups, trainable=mask.bool().tolist(), weights=self.model.values(), initial=self.model.initial.tolist(),
                     self_delta=self.model.self_delta.tolist(), outer_delta=self.model.outer_delta.tolist(), history=self.history)
