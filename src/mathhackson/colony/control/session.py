"""相同外部干预、独立场与个体的实时对照；不预设哪一组获胜。"""
from __future__ import annotations
from ..world import World
from ..protocol import Command, Frame, apply_command, snapshot


class ComparisonFrame(Frame):
    reference: Frame | None = None


class ComparisonSession:
    def __init__(self, seed: int=42, count: int=32, *, warmup: int=220) -> None:
        self.warmup=warmup
        self.world=World(seed,count,warmup)
        self.reference: World | None=None

    @property
    def worlds(self) -> tuple[World, ...]:
        return (self.world,) if self.reference is None else (self.world,self.reference)

    def frame(self) -> ComparisonFrame:
        main=snapshot(self.world)
        reference=snapshot(self.reference) if self.reference is not None else None
        return ComparisonFrame(**main.model_dump(),reference=reference)

    def tick(self) -> None:
        if not self.world.paused:
            for world in self.worlds:
                world.tick()

    def dispatch(self, command: Command) -> None:
        if command.kind=='compare':
            if command.value>.5:
                seed,count=self.world.seed,len(self.world.ants)
                self.world=World(seed,count,self.warmup)
                self.reference=World(seed,count,0,mode='rules')
                self.world.event('notice','已从相同初始状态开启对照；两个信息素场完全隔离')
            else:
                self.reference=None
            return
        if command.kind=='reset':
            comparing=self.reference is not None
            self.world=World(command.seed,command.count,self.warmup)
            self.reference=World(command.seed,command.count,0,mode='rules') if comparing else None
            return
        # 两边先验证、后修改；拒绝一侧时不让另一侧悄悄接受不同实验条件。
        if command.kind in ('wall','food'):
            for world in self.worlds:
                if command.kind=='wall':
                    wall,_,message=world.plan_wall(command.x,command.y,command.hx,command.hy,command.angle)
                    valid=wall is not None
                else:
                    valid,message=world.plan_food(command.x,command.y)
                if not valid:
                    self.world.event('notice',message+'；两侧均未修改' if self.reference else message)
                    return
        if command.kind in ('learning','freeze-ant'):
            apply_command(self.world,command)
            return
        for world in self.worlds:
            apply_command(world,command)
