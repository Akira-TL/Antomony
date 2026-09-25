"""局部行动约束：方向保持、前向触角和停滞脱离；不是学习效果声明。"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from .geometry import Array, ray_distance, unit
if TYPE_CHECKING:
    from .world import Ant, World

DECISION_TICKS = 4
MAX_TURN = .42


def exploration(world: World, ant: Ant, reverse: bool = False) -> Array:
    """只选局部射线已确认可达的位置；目标保持到到达、超时或被新墙阻断。"""
    if ant.waypoint is not None:
        offset = ant.waypoint-ant.position
        distance = float(np.linalg.norm(offset))
        clear = ray_distance(ant.position, offset/max(distance, .01), world.walls,
                             world.half, world.radius, reach=distance)
        if reverse or distance < .35 or world.tick_count >= ant.waypoint_deadline or clear < distance-.02:
            ant.waypoint = None
    if ant.waypoint is None:
        probes = np.stack([unit(ant.heading+float(turn)) for turn in world.turns])
        lengths = np.asarray(ant.rays if len(ant.rays)==len(probes) else [
            ray_distance(ant.position, d, world.walls, world.half, world.radius) for d in probes])
        usable = lengths >= .45
        if not np.any(usable):
            return unit(ant.heading+math.pi)
        # 优先较远净空；只在相近的有效采样之间保留朝向偏好和小扰动。
        scores = lengths + (.8 if reverse else .22)*np.cos(world.turns-(math.pi if reverse else 0.))
        scores += ant.brain.rng.normal(0, .035, len(scores))
        scores[~usable] = -np.inf
        index = int(np.argmax(scores))
        ant.waypoint = (ant.position+probes[index]*min(2.1, float(lengths[index])*.85)).astype(np.float32)
        ant.waypoint_deadline = world.tick_count+30
    offset = ant.waypoint-ant.position
    return offset/max(.01, float(np.linalg.norm(offset)))


def direction(world: World, ant: Ant) -> Array:
    # 距上次实际留下空间标记太久，才重新找有效目标；不是每次碰撞都反转。
    timed_out = world.tick_count-ant.last_mark_tick >= 30 and world.tick_count >= ant.ignore_scent_until
    if timed_out:
        ant.ignore_scent_until = world.tick_count+80
        ant.escape_until = world.tick_count+20
        ant.waypoint = None
    escaping = world.tick_count < ant.escape_until
    if ant.carrying and not escaping:
        norm = float(np.linalg.norm(ant.home))
        aim = ant.home/max(norm, .01)
        reach = min(norm, 2.4)
        if ray_distance(ant.position, aim, world.walls, world.half, world.radius, reach=reach) >= reach-.02:
            return aim
        # 直返被阻断时继续走下方局部标记采样，不能跳过回巢通道。
    visible: list[tuple[float, Array]] = []
    for food in world.foods:
        delta = np.asarray([food.x, food.y], np.float32) - ant.position
        distance = float(np.linalg.norm(delta))
        if food.amount > 0 and 1e-5 < distance < 4.8:
            aim = delta / distance
            if ray_distance(ant.position, aim, world.walls, world.half, 0, reach=distance) >= distance-1e-4:
                visible.append((distance, aim))
    if visible and not escaping and not ant.carrying:
        return min(visible, key=lambda pair: pair[0])[1]
    wanted = exploration(world, ant, timed_out).copy()
    if world.tick_count >= ant.ignore_scent_until:
        angles = ant.heading + np.asarray([-.9, -.45, 0., .45, .9], np.float32)
        probes = np.stack([unit(float(angle)) for angle in angles])
        channel = 0 if ant.carrying else 1
        scent = np.asarray([world.field.sample(ant.position+1.1*d, channel)
                            if ray_distance(ant.position, d, world.walls, world.half, world.radius) >= 1.1 else 0.
                            for d in probes])
        # 嗅探前方，而不是每步追逐任意方向的浓度梯度峰。
        weights = scent / (1. + scent)
        if float(weights.sum()) > .01:
            wanted += .65 * (weights @ probes) / float(weights.sum())
    return wanted.astype(np.float32)


def choose_motion(world: World, ant: Ant, x: Array, positions: Array) -> tuple[float, Array]:
    forward_index = len(world.turns)//2
    emergency = x[forward_index, 2]*2.4 < world.speed*world.dt*1.6 or ant.stuck > 4
    if world.tick_count >= ant.next_decision or (emergency and world.tick_count >= ant.next_emergency):
        wanted = direction(world, ant)
        # 仅用可见近邻的排斥意向，避免把拥挤中心误作值得持续追逐的目标。
        separation = np.zeros(2, np.float32)
        for other in positions:
            delta = ant.position-other
            distance = float(np.linalg.norm(delta))
            if 1e-6 < distance < .9:
                separation += delta/distance*(1-distance/.9)
        norm = float(np.linalg.norm(separation))
        if norm > 0:
            wanted += .65*separation/max(1., norm)
        wanted /= max(1., float(np.linalg.norm(wanted)))
        predicted = ant.brain.predict(x)
        forward = unit(ant.heading)
        right = np.asarray([-forward[1], forward[0]], np.float32)
        local = np.asarray([wanted@forward, wanted@right], np.float32)
        scores = np.clip(predicted[:, :2], -1.3, 1.3)@local - 2.3*np.clip(predicted[:, 2], 0, 1)
        scores += .25*np.cos(world.turns)
        scores += ant.brain.rng.normal(0, .015 if ant.carrying else .025, len(scores))
        index = int(np.argmax(scores))
        ant.action = index
        ant.target_heading = ant.heading + float(world.turns[index])
        ant.next_decision = world.tick_count + DECISION_TICKS
        ant.next_emergency = world.tick_count + 2
        ant.decisions += 1
    error = math.atan2(math.sin(ant.target_heading-ant.heading), math.cos(ant.target_heading-ant.heading))
    turn = float(np.clip(error, -MAX_TURN, MAX_TURN))
    # 学习输入必须对应实际执行的限速转向，不能使用未执行的原始候选角。
    values = x[forward_index].copy()
    angles = np.append(world.turns, np.pi)
    clear = np.append(x[:, 2], x[0, 2])
    values[0:2] = [math.cos(turn), math.sin(turn)]
    values[2] = np.interp(turn, angles, clear)
    values[3] = np.interp(turn-math.pi/6, angles, clear, period=2*math.pi)
    values[4] = np.interp(turn+math.pi/6, angles, clear, period=2*math.pi)
    return ant.heading + turn, values


def deposit_trail(world: World, ant: Ant, actual: Array) -> None:
    """距上个标记足够远才沉积；原地抖动不能靠往返路程反复增强标记。"""
    ant.trail_distance += min(float(np.linalg.norm(actual)), world.speed*world.dt)
    if ant.last_mark_position is None:
        ant.last_mark_position = (ant.position-actual).copy()
    distance = float(np.linalg.norm(ant.position-ant.last_mark_position))
    if distance < .36:
        return
    channel = 1 if ant.carrying else 0
    # 新标记的源龄衰减与环境中已有标记的蒸发是两件不同的事。
    strength = (.42 if ant.carrying else .13)*math.exp(-ant.age/24.)
    world.field.deposit(ant.position, channel, strength)
    ant.last_mark_position = ant.position.copy()
    ant.last_mark_tick = world.tick_count
