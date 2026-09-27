"""单时钟三世界，实际行动、同步编辑和个体学习状态保持隔离。"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from mathhackson.colony.geometry import Wall
from mathhackson.training.comparison.actors import NeuralForager
from mathhackson.training.comparison.continuous import Actor, Condition, ContinuousPlan, make_actors, save_actors
from mathhackson.training.comparison.feedback import learning_feedback
from mathhackson.training.comparison.online_actor import OnlineForager
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.environment import LocalObservation
from mathhackson.training.foraging.trust_candidate import TrustConfig
from .hazards import Trap
from .protocol import (AntView, Control, Counts, Edit, Frame, GroupKey, Notice, ParameterModule,
                       ParameterPoint, ParameterView, Preview, SessionConfig, TrapView, WorldView)
from .recording import RecordStore
from .world import (INTERACTIVE_AWAY_RADIUS, INTERACTIVE_HOME_RADIUS, EditableColony)


@dataclass
class Group:
    key: GroupKey
    world: EditableColony
    actors: list[Actor]
    accepted: int = 0
    decisions: int = 0


def make_plan(config: SessionConfig) -> ContinuousPlan:
    return ContinuousPlan(seeds=(config.seed,), arms=("always", "mlp", "rules"), respawn=True,
        conditions=(Condition(name="reference", disturbance=DisturbanceConfig(signal_strength=0., injury_per_step=0.)),),
        environment=ColonyConfig(ants=config.ants, horizon=config.horizon, stock=config.stock,
            food_distance_min=10., food_distance_span=2., trail_profile="bounded-local-v2", nest_signal_strength=1.,
            nest_signal_radius=3.5, home_radius=INTERACTIVE_HOME_RADIUS, away_radius=INTERACTIVE_AWAY_RADIUS),
        adaptation=TrustConfig(window=4, feedback_mode="observed-window"), feedback_profile="survival-v1",
        policy_directory="models/interactive/memory", gate_directory="models/interactive/gate",
        mlp_directory="models/interactive/mlp", motor="models/interactive/motor/update-001200.npz")


class LiveSession:
    def __init__(self, config: SessionConfig, directory: Path, *, plan: ContinuousPlan | None = None) -> None:
        self.config, self.plan = config, plan or make_plan(config)
        if (self.plan.environment.ants != config.ants or self.plan.environment.horizon != config.horizon
                or self.plan.environment.stock != config.stock or self.plan.seeds != (config.seed,)
                or self.plan.adaptation.window != 4 or self.plan.adaptation.credit_horizon is not None
                or self.plan.feedback_profile != "survival-v1" or not self.plan.respawn
                or self.plan.arms != ("always", "mlp", "rules")):
            raise ValueError("现场配置与实际执行方案不一致")
        self.groups = [Group(key, EditableColony(config.seed, self.plan.environment, rich_scene=True),
                             make_actors(self.plan, config.seed, arm))
                       for key, arm in (("adaptive", "always"), ("mlp", "mlp"), ("rules", "rules"))]
        self.run_id = directory.name
        self.paused, self.learning, self.rate = True, True, 1
        self.step_ms = 0.
        self.checkpoint_tick = -1
        self.notices: deque[Notice] = deque(maxlen=6)
        self.parameter_history: list[deque[ParameterPoint]] = [deque(maxlen=512) for _ in range(config.ants)]
        self.records = RecordStore(directory, config, self.plan)
        self.checkpoint()
        self._remember_parameters()
        self.records.append(self.frame())

    @property
    def tick(self) -> int:
        return self.groups[0].world.steps

    @property
    def done(self) -> bool:
        return False

    def notice(self, message: str) -> None:
        self.notices.appendleft(Notice(tick=self.tick, message=message))

    def _trap(self, edit: Edit, world: EditableColony) -> Trap:
        return Trap(id=max((t.id for t in world.traps), default=-1) + 1,
                    x=edit.x, y=edit.y, born_at=self.tick, radius=edit.radius, injury=edit.injury,
                    speed_multiplier=edit.speed_multiplier, period=edit.period, motion_amplitude=edit.motion_amplitude)

    def preview(self, edit: Edit) -> Preview:
        if self.done:
            return Preview(valid=False, message="本轮已结束，请开始新一轮", tick=self.tick)
        relocations: list[int] = []
        for group in self.groups:
            world = group.world
            if edit.kind == "food":
                error = world.food_placement_error(edit.x, edit.y, edit.stock)
            elif edit.kind == "wall":
                planned, error = world.plan_wall(Wall(0, edit.x, edit.y, edit.hx, edit.hy, edit.angle))
                if planned is not None:
                    current = np.stack([ant.position for ant in world.ants])
                    relocations.append(int(np.count_nonzero(np.linalg.norm(planned - current, axis=1) > 1e-6)))
            elif edit.kind == "trap":
                error = world.trap_placement_error(self._trap(edit, world))
            elif edit.kind == "erase-wall":
                error = None if any(w.id == edit.identifier for w in world.walls) else "墙体不存在"
            elif edit.kind == "erase-trap":
                error = None if any(t.id == edit.identifier for t in world.traps) else "作用区不存在"
            else:
                error = None
            if error:
                title = {"adaptive": "自训练组", "mlp": "MLP组", "rules": "规则组"}[group.key]
                return Preview(valid=False, message=f"{title}：{error}", tick=self.tick)
        if edit.kind == "wall" and any(relocations):
            counts = " / ".join(str(count) for count in relocations)
            return Preview(valid=True, message=f"三组均可放置；将分别移开 {counts} 只蚂蚁", tick=self.tick)
        return Preview(valid=True, message="三组均可放置" if edit.kind in ("food", "wall", "trap") else "三组均可执行", tick=self.tick)

    def edit(self, command: Edit) -> Preview:
        checked = self.preview(command)
        if not checked.valid:
            return checked
        # 同一事件循环内先核验全部世界，期间不推进仿真，再一起修改。
        for group in self.groups:
            world = group.world
            if command.kind == "food":
                world.add_food(command.x, command.y, command.stock)
            elif command.kind == "wall":
                world.add_wall(command.x, command.y, command.hx, command.hy, command.angle)
            elif command.kind == "trap":
                world.add_trap(self._trap(command, world))
            elif command.kind == "erase-wall":
                world.remove_wall(command.identifier)
            elif command.kind == "erase-trap":
                world.remove_trap(command.identifier)
            else:
                world.signals.trails.values.fill(0.)
        self.records.intervention(self.tick, command)
        labels = {"food": "新增食物", "wall": "新增墙体", "trap": "新增作用区", "erase-wall": "拆除墙体",
                  "erase-trap": "移除作用区", "clear-trails": "清空信息素"}
        self.notice(labels[command.kind] + "，已同步三组")
        return checked

    def control(self, command: Control) -> None:
        at_tick = self.tick
        if command.kind == "pause":
            self.paused = command.enabled or self.done
            self.records.flush()
        elif command.kind == "step":
            if not self.paused:
                raise ValueError("单步前请暂停")
            self.advance()
        elif command.kind == "speed":
            self.rate = command.rate
        elif command.kind == "learning":
            self.learning = command.enabled
            self.notice("恢复受限参数写入" if self.learning else "冻结参数；行动继续")
        else:
            self.checkpoint()
            self.notice(f"参数已保存至第{self.checkpoint_tick}步")
        self.records.control(at_tick, command)

    def advance(self) -> Frame:
        if self.done:
            self.paused = True
            return self.frame()
        start = time.perf_counter()
        next_observations: dict[GroupKey, list[LocalObservation]] = {}
        for group in self.groups:
            world = group.world
            for i in world.release_waiting():
                if isinstance(group.actors[i], OnlineForager):
                    group.actors[i].revive()
            active = [not ant.exhausted for ant in world.ants]
            observations = [world.observation(i) for i in range(len(group.actors))]
            actions = [DirectionAction(False, 0., 0.) if not alive else
                       actor.act(obs, sampled=True) if isinstance(actor, NeuralForager) else actor.act(obs)
                       for actor, obs, alive in zip(group.actors, observations, active, strict=True)]
            injury_before = world.injuries.copy()
            events = world.step(actions)
            observed = [world.observation(i) for i in range(len(group.actors))]
            next_observations[group.key] = observed
            for i, actor in enumerate(group.actors):
                if isinstance(actor, OnlineForager) and active[i]:
                    reward = learning_feedback("survival-v1", events[i], active=True,
                        injury_delta=float(world.injuries[i] - injury_before[i]), injury_limit=1.)
                    record = actor.feedback(reward, observed[i], terminal=world.done or world.ants[i].exhausted,
                        tick=world.steps, individual=i, continuing_after_death=world.ants[i].exhausted and not world.done,
                        accept_updates=self.learning)
                    if record:
                        group.decisions += 1
                        group.accepted += int(record.accepted)
                        self.records.update(group.key, record)
        if any(g.world.steps != self.tick for g in self.groups):
            raise AssertionError("三组时钟失配")
        if self.tick % self.plan.checkpoint_every == 0 or self.done:
            self.checkpoint()
        if self.tick % 4 == 0 or self.done:
            self._remember_parameters()
        self.step_ms = (time.perf_counter() - start) * 1000.
        if self.done:
            self.paused = True
        frame = self.frame(next_observations)
        self.records.append(frame)
        return frame

    def _remember_parameters(self) -> None:
        for i, actor in enumerate(self.groups[0].actors):
            self.parameter_history[i].append(ParameterPoint(tick=self.tick, values=actor.agent.weights().tolist()))

    def checkpoint(self) -> None:
        if self.checkpoint_tick == self.tick:
            return
        for group in self.groups:
            folder = self.records.directory / group.key
            folder.mkdir(exist_ok=True)
            save_actors(group.actors, folder, self.tick, include_memory=True)
            with (folder / f"tick-{self.tick:04d}.field.npz").open("xb") as stream:
                np.savez_compressed(stream, values=group.world.signals.trails.values,
                                    blocked=group.world.signals.trails.blocked)
        self.checkpoint_tick = self.tick
        self.records.flush()

    def frame(self, observations: dict[GroupKey, list[LocalObservation]] | None = None) -> Frame:
        views = []
        for group in self.groups:
            world = group.world
            observed = observations.get(group.key) if observations else None
            writes = [actor.agent.writes if isinstance(actor, OnlineForager) else 0 for actor in group.actors]
            counts = Counts(delivered=sum(a.deliveries for a in world.ants), pickups=sum(a.pickups for a in world.ants),
                deaths=int(world.deaths.sum()), exhaustions=int((world.terminations - world.deaths).sum()),
                revivals=int(world.revivals.sum()), active=sum(not a.exhausted for a in world.ants),
                injury=float(world.total_injury.sum()), decisions=group.decisions, accepted=group.accepted,
                writes=sum(writes), stock=world.stock)
            ants = [AntView(id=i, x=float(a.position[0]), y=float(a.position[1]), heading=a.heading,
                carrying=a.carrying, pending=a.exhausted, injury=float(world.injuries[i]),
                exploration=a.exploration_left, reserve=a.reserve_left, delivered=a.deliveries,
                deaths=int(world.deaths[i]), exhaustions=int(world.terminations[i] - world.deaths[i]),
                revivals=int(world.revivals[i]), writes=writes[i],
                receptors=(observed[i] if observed else world.observation(i)).receptors[0].tolist())
                for i, a in enumerate(world.ants)]
            field = np.clip(world.signals.trails.values * 72., 0., 255.).astype(np.uint8)
            views.append(WorldView(key=group.key, counts=counts, ants=ants, walls=world.walls, foods=world.foods,
                traps=[TrapView(spec=t, position=t.position(self.tick).tolist(), active=t.active(self.tick)) for t in world.traps],
                field=base64.b64encode(field.tobytes()).decode()))
        return Frame(run_id=self.run_id, tick=self.tick, horizon=self.config.horizon, seed=self.config.seed,
            paused=self.paused, done=self.done, learning=self.learning, rate=self.rate, step_ms=self.step_ms,
            checkpoint_tick=self.checkpoint_tick, groups=views, notices=list(self.notices))

    def parameters(self, key: GroupKey, individual: int) -> ParameterView:
        if not 0 <= individual < self.config.ants:
            raise ValueError("个体编号无效")
        actor = next(g for g in self.groups if g.key == key).actors[individual]
        modules = []

        def add(name: str, label: str, tensors: list[torch.Tensor]) -> None:
            values = torch.cat([v.detach().flatten() for v in tensors]).tolist()
            modules.append(ParameterModule(key=name, label=label, frozen=True,
                points=[ParameterPoint(tick=0, values=values), ParameterPoint(tick=self.tick, values=values)]))

        if isinstance(actor, OnlineForager):
            modules.append(ParameterModule(key="adaptive", label="方向修正", frozen=not self.learning,
                points=list(self.parameter_history[individual]) + [ParameterPoint(tick=self.tick, values=actor.agent.weights().tolist())]))
            add("base", "基础信号与方向", list(actor.agent.policy.base.parameters()))
            add("recent", "近期记忆", [actor.agent.policy.recent_weights])
            add("sparse", "稀疏记忆", [actor.agent.policy.sparse_weights])
            add("motor", "动作底座", list(actor.agent.motor.parameters()))
            add("decision", "接受判断（当前未启用）", list(actor.controller.model.parameters()))
        elif isinstance(actor, NeuralForager):
            add("base", "普通MLP", list(actor.model.parameters()))
            add("motor", "动作底座", list(actor.motor.parameters()))
        return ParameterView(run_id=self.run_id, tick=self.tick, group=key, individual=individual, modules=modules)

    def close(self) -> None:
        self.checkpoint()
        self.records.close()
