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


def direction(world: World, ant: Ant) -> Array:
    ant.wander = ant.heading + float(ant.brain.rng.normal(0, .30))
    forward = unit(ant.heading)
    if ant.carrying:
        norm = float(np.linalg.norm(ant.home))
        return ant.home / max(norm, .01)
    visible: list[tuple[float, Array]] = []
    for food in world.foods:
        delta = np.asarray([food.x, food.y], np.float32) - ant.position
        distance = float(np.linalg.norm(delta))
        if food.amount > 0 and 1e-5 < distance < 4.8:
            aim = delta / distance
            if ray_distance(ant.position, aim, world.walls, world.half, 0, reach=distance) >= distance-1e-4:
                visible.append((distance, aim))
    if visible:
        return min(visible, key=lambda pair: pair[0])[1]
    # 局部身体历史只用于摆脱反复打转，不读取全局食物位置或虚假信号标签。
    if len(ant.recent_positions) == 40 and world.tick_count >= ant.ignore_scent_until:
        if np.linalg.norm(ant.position-ant.recent_positions[0]) < .9:
            ant.ignore_scent_until = world.tick_count + 80
            ant.wander += (1 if ant.id % 2 else -1) * 1.2
    wanted = .65*forward + .35*unit(ant.wander)
    if world.tick_count >= ant.ignore_scent_until:
        angles = ant.heading + np.asarray([-.9, -.45, 0., .45, .9], np.float32)
        probes = np.stack([unit(float(angle)) for angle in angles])
        scent = np.asarray([world.field.sample(ant.position+1.1*d, 1) for d in probes])
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
    distance = min(float(np.linalg.norm(actual)), world.speed*world.dt)
    ant.trail_distance += distance
    if distance < .01:
        return
    channel = 1 if ant.carrying else 0
    strength = (.55 if ant.carrying else .18)*distance*math.exp(-ant.trail_distance/12.)
    world.field.deposit(ant.position, channel, strength)
