from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from .environment import SingleAntEnvironment
from .model import GROUPS, INPUTS, Decision, SelfModifyingPolicy
from .schemas import Command, EpisodeRecord, GroupState, State


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
        self._begin()

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
        if self.env.done:
            self.finish("到达目标" if self.env.reached else "回合结束", True)

    def command(self, command: Command) -> None:
        action = command.action
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
        groups = [GroupState(id=group, label=label, start=start, end=end,
                             frozen=not bool(self.model.mask()[start].any()), manual=group in self.model.manual_frozen,
                             reason="手动冻结" if group in self.model.manual_frozen else
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
                     groups=groups, weights=self.model.values(), initial=self.model.initial.tolist(),
                     self_delta=self.model.self_delta.tolist(), outer_delta=self.model.outer_delta.tolist(), history=self.history)
