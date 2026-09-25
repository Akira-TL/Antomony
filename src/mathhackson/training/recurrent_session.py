"""独立的单蚁循环记忆试验会话与逐回合记录。"""
from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from mathhackson.colony.geometry import unit

from .environment import SingleAntEnvironment
from .recurrent import Action, RecurrentPolicy
from .schemas import RecurrentCommand, RecurrentEpisode, RecurrentParameterGroup, RecurrentState, Task

GROUP_LABELS = ("基础动作", "观察到隐藏状态", "隐藏状态到隐藏状态", "隐藏状态偏置", "记忆行动修正")
GROUP_IDS = ("motor", "input_weights", "hidden_weights", "hidden_bias", "action_weights")


class RecurrentSession:
    def __init__(self, directory: Path, seed: int = 20260926) -> None:
        self.id = uuid4().hex[:12]
        self.directory = directory / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.model = RecurrentPolicy(seed)
        self.env = SingleAntEnvironment(seed + 1)
        self.task: Task = "normal"
        self.paused = True
        self.error = ""
        self.speed = 1
        self.tick = 0
        self.episode = 1
        self.baseline = 0.
        self.history: list[RecurrentEpisode] = []
        self._begin()

    def _begin(self) -> None:
        self.env.reset()
        self.model.reset_state()
        self.actions: list[Action] = []
        self.rewards: list[float] = []
        self.observations: list[np.ndarray] = []
        self.hidden_states: list[np.ndarray] = []
        if self.task == "normal" or (self.task == "mixed" and self.rng.random() < .5):
            self.perturbation = 0.
        else:
            self.perturbation = float(self.rng.choice([-1., 1.]) * .25)

    def motor_alignment(self) -> tuple[int, int]:
        probe = RecurrentPolicy(0)
        with torch.no_grad():
            probe.motor.copy_(self.model.motor.detach())
        probe.phase = "autonomous"
        reached = [0, 0]
        for index in range(12):
            environment = SingleAntEnvironment(index)
            environment.target = unit(-math.pi + (index + .5) * math.pi / 6.) * 4.
            environment.distance = 4.
            while not environment.done:
                decision = probe.decide(environment.observation())
                environment.step(decision.move, decision.turn)
                probe.observe_result(environment.observation())
            reached[index // 6] += int(environment.reached)
            probe.reset_state()
        return reached[0], reached[1]

    def finish(self, reason: str, optimize: bool) -> None:
        if self.actions:
            self.model.finish(self.actions, self.rewards, self.baseline, optimize)
            filename = f"episode-{self.episode:06d}.npz"
            np.savez_compressed(
                self.directory / filename,
                motor=self.model.motor.detach().numpy(),
                input_weights=self.model.input_weights.detach().numpy(),
                hidden_weights=self.model.hidden_weights.detach().numpy(),
                hidden_bias=self.model.hidden_bias.detach().numpy(),
                action_weights=self.model.action_weights.detach().numpy(),
                observations=np.asarray(self.observations),
                hidden_states=np.asarray(self.hidden_states),
                rewards=np.asarray(self.rewards),
                actions=np.asarray([(action.move, action.turn) for action in self.actions]),
                perturbation=self.perturbation,
            )
            record = RecurrentEpisode(episode=self.episode, phase=self.model.phase, task=self.task,
                                      perturbation=self.perturbation, reached=self.env.reached,
                                      steps=len(self.actions), reward=sum(self.rewards), checkpoint=filename)
            (self.directory / filename.replace(".npz", ".json")).write_text(
                record.model_dump_json(indent=2), encoding="utf-8")
            self.history.append(record)
            self.history = self.history[-100:]
            self.episode += 1
            if optimize:
                self.baseline = .9 * self.baseline + .1 * (sum(self.rewards) / len(self.rewards))
        self._begin()

    def step(self) -> None:
        if self.env.steps == 8:
            self.env.turn_bias = self.perturbation
        observation = self.env.observation()
        decision = self.model.decide(observation)
        reward = self.env.step(decision.move, decision.turn)
        self.model.observe_result(self.env.observation())
        self.actions.append(decision)
        self.rewards.append(reward)
        self.observations.append(observation)
        self.hidden_states.append(self.model.hidden.detach().numpy().copy())
        self.tick += 1
        if self.env.done:
            self.finish("到达目标" if self.env.reached else "回合结束", True)

    def command(self, command: RecurrentCommand) -> None:
        if command.action == "phase" and command.phase == "memory" and self.model.phase == "motor":
            left, right = self.motor_alignment()
            if left < 5 or right < 5:
                raise ValueError(f"基础动作尚未通过定向检查：两侧分别触达 {left}/6、{right}/6")
        if command.action == "pause":
            self.paused = True
        elif command.action == "play":
            if self.error:
                raise ValueError(self.error)
            self.paused = False
        elif command.action == "step":
            self.paused = True
            if self.error:
                raise ValueError(self.error)
            self.step()
        elif command.action == "speed":
            self.speed = command.speed
        else:
            self.paused = True
            self.finish("配置改变，未执行外部更新", False)
            if command.action == "phase":
                self.model.phase = command.phase
            elif command.action == "task":
                self.task = command.task
            self._begin()

    def state(self) -> RecurrentState:
        action = self.actions[-1] if self.actions else None
        masks = self.model.trainable_masks()
        groups = [RecurrentParameterGroup(id=identity, label=label, values=value,
                                          changes=change.detach().reshape(change.shape[0], -1).tolist(),
                                          trainable=mask.bool().reshape(mask.shape[0], -1).tolist())
                  for identity, label, value, change, mask in zip(
                      GROUP_IDS, GROUP_LABELS, self.model.values(), self.model.outer_delta,
                      masks, strict=True)]
        return RecurrentState(session=self.id, paused=self.paused, error=self.error,
                     phase=self.model.phase, task=self.task, speed=self.speed,
                     tick=self.tick, episode=self.episode, steps=self.env.steps,
                     horizon=self.env.horizon, x=float(self.env.position[0]),
                     y=float(self.env.position[1]), heading=self.env.heading,
                     target_x=float(self.env.target[0]), target_y=float(self.env.target[1]),
                     distance=self.env.distance, max_turn=self.env.max_turn,
                     move=action.move if action else False, turn=action.turn if action else 0.,
                     move_probability=action.move_probability if action else 0.,
                     reward=self.env.reward, total_reward=sum(self.rewards),
                     reached=self.env.reached, perturbation=self.perturbation,
                     active_perturbation=self.env.turn_bias,
                     outer_updates=self.model.outer_updates,
                     hidden=self.model.hidden.detach().tolist(), groups=groups, history=self.history)
