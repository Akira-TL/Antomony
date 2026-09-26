"""交互世界的资源库存；复用训练环境的动作、反馈及复活语义。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mathhackson.colony.geometry import Wall
from mathhackson.training.direction.policy import DirectionAction
from mathhackson.training.foraging.colony import ColonyConfig, ColonyInteraction
from mathhackson.training.foraging.disturbance import DisturbanceConfig
from mathhackson.training.foraging.revival import RevivingColony
from mathhackson.training.foraging.signals import SignalSource
from .signals import BarrierSignals


@dataclass
class Food:
    id: int
    x: float
    y: float
    stock: int


class EditableColony(RevivingColony):
    def __init__(self, seed: int, config: ColonyConfig) -> None:
        super().__init__(seed, config, DisturbanceConfig(signal_strength=0., injury_per_step=0.))
        self.foods = [Food(0, float(self.food[0]), float(self.food[1]), self.stock)]
        self.food_origins: list[int | None] = [None] * len(self.ants)
        self.supplied_stock = self.stock
        signals = BarrierSignals(config.trail_profile)
        signals.trails.values[:] = self.signals.trails.values
        self.signals = signals

    @property
    def done(self) -> bool:
        # 现场允许库存耗尽后继续添加；只有本次明确的时限结束运行。
        return self.steps >= self.config.horizon

    def food_sources(self) -> tuple[SignalSource, ...]:
        return tuple(SignalSource(food.x, food.y, self.config.food_radius, self.config.food_strength,
                                  (1., 0., 0., 0., 0., 0., 0., 0.)) for food in self.foods if food.stock)

    def take_food(self, index: int) -> bool:
        ant = self.ants[index]
        available = [(float(np.linalg.norm(ant.position - np.asarray((food.x, food.y), dtype=np.float32))), food)
                     for food in self.foods if food.stock]
        close = [(distance, food) for distance, food in available if distance < .4]
        if not close:
            return False
        _, food = min(close, key=lambda pair: (pair[0], pair[1].id))
        food.stock -= 1
        self.stock -= 1
        self.food_origins[index] = food.id
        return True

    def return_food(self, index: int) -> None:
        origin = self.food_origins[index]
        food = next((food for food in self.foods if food.id == origin), None)
        if food is None:
            raise ValueError("携带食物没有可追溯来源")
        food.stock += 1
        self.stock += 1
        self.food_origins[index] = None

    def food_placement_error(self, x: float, y: float, stock: int) -> str | None:
        if not math.isfinite(x) or not math.isfinite(y) or type(stock) is not int or not 1 <= stock <= 4096:
            return "食物位置或库存无效"
        position = np.asarray([x, y], dtype=np.float32)
        if np.any(np.abs(position) + .4 > self.signals.trails.half):
            return "食物超出场地"
        if np.linalg.norm(position - self.home) < 1.1:
            return "食物不能覆盖巢穴"
        if any(np.linalg.norm(position - (f.x, f.y)) < .8 for f in self.foods):
            return "食物位置重叠"
        if any(wall.overlaps(position, .4) for wall in self.walls):
            return "食物不能覆盖墙体"
        if len(self.foods) >= 64:
            return "食物源已达到64个上限"
        return None

    def add_food(self, x: float, y: float, stock: int) -> Food:
        error = self.food_placement_error(x, y, stock)
        if error:
            raise ValueError(error)
        food = Food(len(self.foods), x, y, stock)
        self.foods.append(food)
        self.stock += stock
        self.supplied_stock += stock
        return food

    def wall_placement_error(self, wall: Wall) -> str | None:
        if (not all(math.isfinite(v) for v in (wall.x, wall.y, wall.hx, wall.hy, wall.angle))
                or not .2 <= wall.hx <= 4. or not .2 <= wall.hy <= 4.):
            return "墙体尺寸或位置无效"
        if np.any(np.abs((wall.x, wall.y)) + wall.extent > self.signals.trails.half):
            return "墙体超出场地"
        if wall.overlaps(self.home, 1.):
            return "墙体不能覆盖巢穴"
        if any(wall.overlaps(np.asarray([f.x, f.y]), .4) for f in self.foods):
            return "墙体不能覆盖食物"
        if any(wall.intersects(other) for other in self.walls):
            return "墙体位置重叠"
        if any(not ant.exhausted and wall.overlaps(ant.position, .18) for ant in self.ants):
            return "墙体覆盖活动个体"
        if len(self.walls) >= 32:
            return "墙体已达到32个上限"
        return None

    def add_wall(self, x: float, y: float, hx: float, hy: float, angle: float = 0.) -> Wall:
        wall = Wall(max((w.id for w in self.walls), default=-1) + 1, x, y, hx, hy, angle)
        error = self.wall_placement_error(wall)
        if error:
            raise ValueError(error)
        self.walls.append(wall)
        self.signals.trails.set_walls(self.walls)
        return wall

    def remove_wall(self, identifier: int) -> None:
        found = next((wall for wall in self.walls if wall.id == identifier), None)
        if found is None:
            raise ValueError("墙体不存在")
        self.walls.remove(found)
        self.signals.trails.set_walls(self.walls)

    def step(self, actions: list[DirectionAction]) -> list[ColonyInteraction]:
        events = super().step(actions)
        for i, event in enumerate(events):
            if event.delivered:
                self.food_origins[i] = None
        return events
