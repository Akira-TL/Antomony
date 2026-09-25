"""可校验的浏览器命令与只读快照。"""
from __future__ import annotations
import base64
from dataclasses import asdict
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .world import World

class Command(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    kind: Literal['pause','step','reset','speed','wall','erase','food','scent','clear','learning','freeze-ant','wind','fields']
    x: float=Field(default=0,ge=-14,le=14)
    y: float=Field(default=0,ge=-10,le=10)
    value: float=Field(default=0,ge=-4,le=4)
    ant: int=Field(default=0,ge=0,le=63)
    seed: int=Field(default=42,ge=0,lt=2**31)
    count: int=Field(default=32,ge=4,le=64)
    hx: float=Field(default=.4,ge=.25,le=4)
    hy: float=Field(default=2,ge=.25,le=4)
    angle: float=Field(default=0,ge=-1000,le=1000)

class AntView(BaseModel):
    id: int
    x: float
    y: float
    heading: float
    carrying: bool
    delivered: int
    contacts: int
    updates: int
    frozen: bool
    error: float
    delta: float
    drift: float
    fingerprint: str
    birth_loss: float
    warm_loss: float
    hidden: list[float]
    inputs: list[float]
    prediction: list[float]
    rays: list[float]
    action: int
    sense_x: float
    sense_y: float
    sense_heading: float

class WallView(BaseModel):
    id: int
    x: float
    y: float
    hx: float
    hy: float
    angle: float = 0.

class FoodView(BaseModel):
    id: int
    x: float
    y: float
    amount: int

class EventView(BaseModel):
    tick: int
    kind: str
    message: str
    ant: int
    value: float

class Frame(BaseModel):
    tick: int
    seconds: float
    seed: int
    paused: bool
    rate: int
    delivered: int
    contacts: int
    samples: int
    mean_error: float
    tick_ms: float
    wind: float
    field_enabled: bool
    field_width: int
    field_height: int
    pheromones: str
    ants: list[AntView]
    walls: list[WallView]
    foods: list[FoodView]
    events: list[EventView]


def snapshot(w: World) -> Frame:
    ants=[AntView(id=a.id,x=float(a.position[0]),y=float(a.position[1]),heading=a.heading,carrying=a.carrying,delivered=a.delivered,contacts=a.contacts,updates=a.brain.updates,frozen=a.brain.frozen,error=a.brain.last_loss,delta=a.brain.last_delta,drift=a.brain.drift,fingerprint=a.brain.fingerprint(),birth_loss=a.brain.birth_loss,warm_loss=a.brain.warm_loss,hidden=a.brain.hidden.tolist(),inputs=a.inputs.tolist(),prediction=a.brain.output.tolist(),rays=a.rays,action=a.action,sense_x=a.sense_x,sense_y=a.sense_y,sense_heading=a.sense_heading) for a in w.ants]
    field=np.clip(w.field.values*72,0,255).astype(np.uint8)
    return Frame(tick=w.tick_count,seconds=round(w.tick_count*w.dt,1),seed=w.seed,paused=w.paused,rate=w.rate,delivered=sum(a.delivered for a in w.ants),contacts=w.contact_count,samples=w.samples,mean_error=w.error_sum/max(1,w.samples),tick_ms=round(w.last_ms,2),wind=w.wind,field_enabled=w.field.enabled,field_width=w.field.width,field_height=w.field.height,pheromones=base64.b64encode(field.tobytes()).decode(),ants=ants,walls=[WallView(**asdict(x)) for x in w.walls],foods=[FoodView(**asdict(x)) for x in w.foods if x.amount>0],events=[EventView(**asdict(e)) for e in w.events])


def apply_command(w: World,c: Command) -> World:
    if c.kind=='reset': return World(c.seed,c.count)
    if c.kind=='pause': w.paused=not w.paused
    elif c.kind=='step':
        if w.paused: w.tick()
    elif c.kind=='speed': w.rate=max(1,min(4,int(c.value)))
    elif c.kind=='wall': w.event('notice',w.add_wall(c.x,c.y,c.hx,c.hy,c.angle))
    elif c.kind=='erase': w.event('notice',w.remove_wall(c.x,c.y))
    elif c.kind=='food': w.event('notice',w.add_food(c.x,c.y))
    elif c.kind=='scent':
        w.field.spray(np.asarray([c.x,c.y],np.float32)); w.event('scent','人为喷洒局部食物信号；未通知模型真假')
    elif c.kind=='clear': w.field.values.fill(0); w.event('clear','信息素已清空；模型参数保持不变')
    elif c.kind=='learning':
        freeze=c.value<.5
        for a in w.ants: a.brain.frozen=freeze
        w.event('freeze','全部参数冻结；感知和行动继续' if freeze else '恢复个体在线学习')
    elif c.kind=='freeze-ant' and c.ant<len(w.ants):
        a=w.ants[c.ant]; a.brain.frozen=not a.brain.frozen; w.event('freeze',f'个体 {c.ant:02d} '+('已冻结' if a.brain.frozen else '继续学习'),c.ant)
    elif c.kind=='wind': w.wind=float(c.value); w.event('wind','已改变中央区域的外部横向力')
    elif c.kind=='fields': w.field.enabled=c.value>.5
    return w
