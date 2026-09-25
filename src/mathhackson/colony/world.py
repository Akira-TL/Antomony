"""局部感知蚁群；规则负责搬运与信号，神经预测参与方向选择。"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import math
import time
import numpy as np
from .geometry import Array, Wall, move_discs, ray_distance, unit, relocate_for_wall
from .pheromone import Pheromones
from .brain import Brain
from .navigation import choose_motion, deposit_trail, direction

@dataclass
class Food:
    id: int
    x: float
    y: float
    amount: int=180

@dataclass
class Ant:
    id: int
    position: Array
    heading: float
    brain: Brain
    home: Array
    carrying: bool=False
    following_trail: bool=False
    delivered: int=0
    contacts: int=0
    age: float=0.
    stuck: int=0
    wander: float=0.
    motion: Array=field(default_factory=lambda:np.zeros(2,np.float32))
    inputs: Array=field(default_factory=lambda:np.zeros(8,np.float32))
    rays: list[float]=field(default_factory=list)
    action: int=0
    target_heading: float=0.
    next_decision: int=0
    next_emergency: int=0
    decisions: int=0
    trail_distance: float=0.
    last_mark_position: Array|None=None
    last_mark_tick: int=0
    waypoint: Array|None=None
    waypoint_deadline: int=0
    escape_until: int=0
    ignore_scent_until: int=0
    recent_positions: deque[Array]=field(default_factory=lambda:deque(maxlen=40))
    sense_x: float=0.
    sense_y: float=0.
    sense_heading: float=0.

@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    message: str
    ant: int=-1
    value: float=0.

class World:
    radius=.20
    dt=.1
    speed=1.8
    half=np.asarray([14.,10.],dtype=np.float32)
    turns=np.linspace(-np.pi,np.pi,12,endpoint=False).astype(np.float32)

    def __init__(self,seed: int=42,count: int=32,warmup: int=220) -> None:
        if not 1<=count<=64: raise ValueError('个体数量必须在1到64之间')
        if not 0<=seed<2**31: raise ValueError('种子超出范围')
        self.seed=seed; self.tick_count=0; self.paused=False; self.rate=1
        self.nest=np.asarray([-10.,0.],np.float32)
        self.walls=[Wall(1,-2.5,-4.,.4,2.2),Wall(2,2.,3.,.4,2.2)]
        self.foods=[Food(1,7.,-4.),Food(2,8.,4.),Food(3,1.,7.)]
        self.field=Pheromones(); self.field.set_walls(self.walls)
        self.ants:list[Ant]=[]
        for i in range(count):
            p=self.nest+np.asarray([((i%8)-3.5)*.49,((i//8)-(math.ceil(count/8)-1)/2)*.49],np.float32)
            heading=float(i*2.39996323)
            self.ants.append(Ant(i,p,heading,Brain(seed*1009+i*97+3,warmup),(self.nest-p).astype(np.float32),wander=heading,last_mark_position=p.copy()))
        self.events:deque[Event]=deque(maxlen=24)
        self.samples=0; self.error_sum=0.; self.last_ms=0.; self.contact_count=0
        self.wind=0.; self.pheromone_visible=True
        self.event('ready','独立预热完成；开始局部探索')

    def event(self,kind: str,message: str,ant: int=-1,value: float=0.) -> None:
        self.events.appendleft(Event(self.tick_count,kind,message,ant,value))

    def direction(self,ant: Ant) -> Array:
        return direction(self,ant)

    def sense(self,ant: Ant,positions: Array) -> Array:
        ant.sense_x=float(ant.position[0]); ant.sense_y=float(ant.position[1]); ant.sense_heading=ant.heading
        directions=np.stack([unit(ant.heading+float(turn)) for turn in self.turns])
        clearance=[]
        for d in directions:
            length=ray_distance(ant.position,d,self.walls,self.half,self.radius)
            for i,p in enumerate(positions):
                if i==ant.id: continue
                delta=p-ant.position; along=float(delta@d)
                perp2=float(delta@delta)-along*along
                if along>0 and perp2<(self.radius*2)**2:
                    hit=along-math.sqrt(max(0,(self.radius*2)**2-perp2))
                    length=min(length,max(0.,hit))
            clearance.append(length/2.4)
        c=np.asarray(clearance,np.float32); ant.rays=[float(v*2.4) for v in c]
        x=np.zeros((len(self.turns),8),np.float32)
        x[:,0]=np.cos(self.turns); x[:,1]=np.sin(self.turns); x[:,2]=c
        x[:,3]=np.roll(c,1); x[:,4]=np.roll(c,-1); x[:,5]=float(ant.carrying)
        forward=unit(ant.heading); right=np.asarray([-forward[1],forward[0]],np.float32)
        x[:,6]=float(ant.motion@forward); x[:,7]=float(ant.motion@right)
        return x

    def tick(self) -> None:
        start=time.perf_counter(); positions=np.stack([a.position for a in self.ants])
        motions=[]; chosen=[]; predictions=[]; headings=[]
        for ant in self.ants:
            x=self.sense(ant,positions)
            angle,features=choose_motion(self,ant,x,positions)
            ant.inputs=features.copy()
            predictions.append(ant.brain.inspect(features)); chosen.append(features)
            headings.append(angle)
            displacement=unit(angle)*self.speed*self.dt
            # 外力只改变物理世界；不向模型发送干预类别或正确补偿量。
            if abs(float(ant.position[0]))<5: displacement[1]+=self.wind*self.dt
            motions.append(displacement)
        moved,contacts=move_discs(positions,np.asarray(motions,np.float32),self.radius,self.walls,self.half)
        for i,ant in enumerate(self.ants):
            actual=moved[i]-positions[i]; forward=unit(ant.heading); right=np.asarray([-forward[1],forward[0]],np.float32)
            target=np.asarray([actual@forward/(self.speed*self.dt),actual@right/(self.speed*self.dt),float(contacts[i])],np.float32)
            loss=float(np.mean((predictions[i]-target)**2)); self.error_sum+=loss; self.samples+=1
            ant.brain.last_loss=loss
            if not ant.brain.frozen:
                ant.brain.train(chosen[i],target)
                if ant.id==0 and self.tick_count%20==0:
                    self.event('learn',f'个体 {i:02d} 更新运动预测参数',i,ant.brain.last_delta)
            ant.position=moved[i]; ant.home-=actual; ant.motion=actual/(self.speed*self.dt)
            ant.heading=headings[i]; ant.age+=self.dt
            ant.stuck=ant.stuck+1 if float(np.linalg.norm(actual))<.035 else max(0,ant.stuck-1)
            if contacts[i]: ant.contacts+=1; self.contact_count+=1
            ant.recent_positions.append(ant.position.copy())
            self.collect(ant)
            deposit_trail(self,ant,actual)
        self.field.tick(self.dt); self.tick_count+=1
        self.last_ms=(time.perf_counter()-start)*1000

    def collect(self,ant: Ant) -> None:
        if ant.carrying and np.linalg.norm(ant.position-self.nest)<1.5:
            ant.carrying=False; ant.delivered+=1; ant.age=0.; ant.home=self.nest-ant.position
            ant.wander=ant.heading+math.pi; ant.target_heading=ant.wander; ant.next_decision=0
            ant.trail_distance=0.; ant.recent_positions.clear(); ant.waypoint=None
            ant.last_mark_position=ant.position.copy(); ant.last_mark_tick=self.tick_count
            self.event('delivery',f'个体 {ant.id:02d} 搬回一个像素块',ant.id)
        elif not ant.carrying:
            if np.linalg.norm(ant.position-self.nest)<1.5:
                ant.age=0.
            for f in self.foods:
                if f.amount>0 and math.hypot(float(ant.position[0])-f.x,float(ant.position[1])-f.y)<.8:
                    f.amount-=1; ant.carrying=True; ant.age=0; ant.next_decision=0
                    ant.trail_distance=0.; ant.recent_positions.clear(); ant.waypoint=None
                    ant.last_mark_position=ant.position.copy(); ant.last_mark_tick=self.tick_count
                    self.event('pickup',f'个体 {ant.id:02d} 发现资源',ant.id); break

    def plan_wall(self,x: float,y: float,hx: float=.4,hy: float=2.,angle: float=0.) -> tuple[Wall|None,Array|None,str]:
        if not all(math.isfinite(v) for v in (x,y,hx,hy,angle)): return None,None,'坐标必须有限'
        if len(self.walls)>=24: return None,None,'最多24道墙'
        hx=max(.25,min(4.,hx)); hy=max(.25,min(4.,hy))
        wall=Wall(max((w.id for w in self.walls),default=0)+1,x,y,hx,hy,angle)
        if np.any(np.abs([x,y])+wall.extent>=self.half): return None,None,'墙超出场地边界'
        if wall.overlaps(self.nest,1.8): return None,None,'与巢穴重叠，不能覆盖'
        if any(wall.overlaps(np.asarray([f.x,f.y],np.float32),1.) for f in self.foods): return None,None,'与资源点重叠，不能覆盖'
        if any(wall.intersects(v) for v in self.walls): return None,None,'与已有墙体重叠'
        positions=relocate_for_wall(np.stack([a.position for a in self.ants]),wall,self.walls,self.half,self.radius)
        if positions is None: return None,None,'墙边没有足够空间安置个体'
        return wall,positions,'可放置'

    def add_wall(self,x: float,y: float,hx: float=.4,hy: float=2.,angle: float=0.) -> str:
        wall,positions,message=self.plan_wall(x,y,hx,hy,angle)
        if wall is None or positions is None: return message+'，已拒绝'
        displaced=0
        for ant,position in zip(self.ants,positions,strict=True):
            delta=position-ant.position
            if float(np.linalg.norm(delta))>1e-6:
                ant.home-=delta; ant.position=position.copy(); ant.motion.fill(0)
                ant.next_decision=0; ant.recent_positions.clear(); ant.waypoint=None
                ant.last_mark_position=position.copy(); ant.last_mark_tick=self.tick_count; displaced+=1
        self.walls.append(wall); self.field.set_walls(self.walls)
        self.event('wall',f'已加入真实碰撞墙；就近移开 {displaced} 只个体（编辑，不训练）')
        return '墙已放置' if not displaced else f'墙已放置；已就近移开 {displaced} 只个体'

    def remove_wall(self,x: float,y: float) -> str:
        for w in self.walls:
            if w.overlaps(np.asarray([x,y],np.float32), .5):
                self.walls.remove(w); self.field.set_walls(self.walls); self.event('wall','已移除墙体'); return '墙已移除'
        return '这里没有墙'

    def add_food(self,x: float,y: float) -> str:
        if abs(x)>12.5 or abs(y)>8.5: return '资源超出场地'
        if len(self.foods)>=12: return '最多12个资源点'
        if any(w.overlaps(np.asarray([x,y],np.float32),.9) for w in self.walls): return '资源与墙重叠'
        if np.linalg.norm(np.asarray([x,y])-self.nest)<2.5: return '资源不能放在巢穴内'
        self.foods.append(Food(max(f.id for f in self.foods)+1,x,y)); self.event('food','已放置新资源点'); return '资源已放置'
