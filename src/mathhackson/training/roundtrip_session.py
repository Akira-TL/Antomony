"""隔离的双通道往返训练会话，按完整回合记录慢参数与反馈。"""
from __future__ import annotations

import base64
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .recurrent import Action, Write
from .recurrent_session import GROUP_IDS, GROUP_LABELS
from .roundtrip_environment import RoundTripEnvironment
from .roundtrip_policy import RoundTripPolicy
from .schemas import (RecurrentParameterGroup, RoundTripCommand, RoundTripEpisode,
                      RoundTripState)

ROUNDTRIP_GROUP_IDS = (*GROUP_IDS, "release_weights", "release_write_weights")
ROUNDTRIP_GROUP_LABELS = (*GROUP_LABELS, "双通道释放", "释放临时修正")


class RoundTripSession:
    def __init__(self, directory: Path, foundation: Path, seed: int = 20260926) -> None:
        self.id = uuid4().hex[:12]
        self.directory = directory / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = RoundTripPolicy(seed, foundation)
        self.env = RoundTripEnvironment(seed + 1)
        self.paused = True
        self.error = ""
        self.speed = 1
        self.tick = 0
        self.episode = 1
        self.baseline = 0.
        self.history: list[RoundTripEpisode] = []
        self._begin()

    def _begin(self) -> None:
        self.env.reset()
        self.model.reset_state()
        self.actions: list[Action] = []
        self.writes: list[Write] = []
        self.rewards: list[float] = []
        self.observations: list[np.ndarray] = []
        self.hidden_states: list[np.ndarray] = []
        self.fast_states: list[np.ndarray] = []
        self.release_fast_states: list[np.ndarray] = []
        self.start_self = self.model.self_updates

    def finish(self, optimize: bool) -> None:
        if self.actions:
            total = sum(self.rewards)
            self.model.finish(self.actions, self.writes, self.rewards, self.baseline, optimize)
            filename = f"episode-{self.episode:06d}.npz"
            np.savez_compressed(
                self.directory / filename,
                model_version="roundtrip-v1",
                **{identity: parameter.detach().numpy() for identity, parameter in
                   zip(ROUNDTRIP_GROUP_IDS, self.model.parameters, strict=True)},
                observations=np.asarray(self.observations),
                hidden_states=np.asarray(self.hidden_states),
                fast_states=np.asarray(self.fast_states),
                release_fast_states=np.asarray(self.release_fast_states),
                rewards=np.asarray(self.rewards),
                actions=np.asarray([(action.move, action.turn, action.release_home, action.release_food)
                                    for action in self.actions]),
                gates=np.asarray([(write.requested, write.wrote, write.probability)
                                  for write in self.writes]),
                pickups=self.env.pickups,
                delivered=self.env.delivered,
            )
            record = RoundTripEpisode(
                episode=self.episode, phase=self.model.phase, completed=self.env.done,
                steps=len(self.actions),
                pickups=self.env.pickups, delivered=self.env.delivered, reward=total,
                writes=self.model.self_updates - self.start_self, checkpoint=filename)
            (self.directory / filename.replace(".npz", ".json")).write_text(
                record.model_dump_json(indent=2), encoding="utf-8")
            self.history.append(record)
            self.history = self.history[-100:]
            self.episode += 1
            if optimize:
                self.baseline = .9 * self.baseline + .1 * (total / len(self.rewards))
        self._begin()

    def step(self) -> None:
        observation = self.env.observation()
        action = self.model.decide(observation)
        reward = self.env.step(action.move, action.turn, action.release_home, action.release_food)
        write = self.model.observe_result(self.env.observation(), terminal=self.env.done)
        self.actions.append(action)
        self.writes.append(write)
        self.rewards.append(reward)
        self.observations.append(observation)
        self.hidden_states.append(self.model.hidden.detach().numpy().copy())
        self.fast_states.append(self.model.fast.detach().numpy().copy())
        self.release_fast_states.append(self.model.release_fast.detach().numpy().copy())
        self.tick += 1
        if self.env.done:
            self.finish(self.model.phase != "autonomous")

    def command(self, command: RoundTripCommand) -> None:
        if command.action == "write_mode" and self.model.phase != "autonomous":
            raise ValueError("只可在停止外部训练后切换写入对照")
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
            self.finish(False)
            if command.action == "phase":
                self.model.phase = command.phase
                self.model.write_mode = "learned" if command.phase == "adaptive" else "off"
            elif command.action == "write_mode":
                self.model.write_mode = command.write_mode

    def state(self) -> RoundTripState:
        action = self.actions[-1] if self.actions else None
        write = self.writes[-1] if self.writes else None
        masks = self.model.trainable_masks()
        groups = [RecurrentParameterGroup(
            id=identity, label=label, values=value,
            changes=change.detach().reshape(change.shape[0], -1).tolist(),
            trainable=mask.bool().reshape(mask.shape[0], -1).tolist())
            for identity, label, value, change, mask in zip(
                ROUNDTRIP_GROUP_IDS, ROUNDTRIP_GROUP_LABELS, self.model.values(),
                self.model.outer_delta, masks, strict=True)]
        field = np.clip(self.env.field.values * 72, 0, 255).astype(np.uint8)
        observation = self.env.observation()
        return RoundTripState(
            session=self.id, paused=self.paused, error=self.error,
            phase=self.model.phase, write_mode=self.model.write_mode, speed=self.speed,
            tick=self.tick, episode=self.episode, steps=self.env.steps,
            horizon=self.env.horizon, x=float(self.env.position[0]),
            y=float(self.env.position[1]), heading=self.env.heading,
            food_x=float(self.env.food[0]), food_y=float(self.env.food[1]),
            carrying=self.env.carrying, pickups=self.env.pickups, delivered=self.env.delivered,
            move=action.move if action else False, turn=action.turn if action else 0.,
            move_probability=action.move_probability if action else 0.,
            release_home=action.release_home if action else False,
            release_food=action.release_food if action else False,
            release_home_probability=action.release_home_probability if action else 0.,
            release_food_probability=action.release_food_probability if action else 0.,
            home_scent=float(observation[8]), food_scent=float(observation[11]),
            reward=self.env.reward, total_reward=sum(self.rewards),
            write_probability=write.probability if write else 0.,
            write_status="尚未推理" if write is None else "已写入" if write.wrote else
            "选择跳过" if self.model.write_mode == "learned" else
            "写入关闭" if self.model.write_mode == "off" else "幅度为零",
            self_updates=self.model.self_updates, outer_updates=self.model.outer_updates,
            hidden=self.model.hidden.detach().tolist(), fast=self.model.fast.detach().tolist(),
            release_fast=self.model.release_fast.detach().tolist(), groups=groups,
            field_width=self.env.field.width, field_height=self.env.field.height,
            pheromones=base64.b64encode(field.tobytes()).decode(), history=self.history)
