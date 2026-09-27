"""两通道信息素；无通量墙界，有限差分不跨边界环绕。"""
from __future__ import annotations
import numpy as np
from .geometry import Array, Wall

class Pheromones:
    def __init__(self, width: int = 96, height: int = 64, half: tuple[float, float] = (14., 10.)) -> None:
        self.width,self.height=width,height
        self.half=np.asarray(half,dtype=np.float32)
        self.values=np.zeros((2,height,width),dtype=np.float32)
        self.blocked=np.zeros((height,width),dtype=np.bool_)
        self.enabled=True

    def cell(self,p: Array) -> tuple[int,int]:
        x=int(np.clip((p[0]+self.half[0])/(2*self.half[0])*self.width,0,self.width-1))
        y=int(np.clip((p[1]+self.half[1])/(2*self.half[1])*self.height,0,self.height-1))
        return y,x

    def set_walls(self,walls: list[Wall]) -> None:
        x=np.linspace(-self.half[0]+self.half[0]/self.width,self.half[0]-self.half[0]/self.width,self.width)
        y=np.linspace(-self.half[1]+self.half[1]/self.height,self.half[1]-self.half[1]/self.height,self.height)
        self.blocked.fill(False)
        for w in walls:
            c, sn = np.cos(w.angle), np.sin(w.angle)
            dx, dy = x[None,:]-w.x, y[:,None]-w.y
            # 单元在墙局部轴上的投影半径：斜边不泄漏，也不以大外接框遮掉通路。
            ex = abs(c)*self.half[0]/self.width+abs(sn)*self.half[1]/self.height
            ey = abs(sn)*self.half[0]/self.width+abs(c)*self.half[1]/self.height
            self.blocked |= (np.abs(c*dx+sn*dy)<=w.hx+ex)&(np.abs(-sn*dx+c*dy)<=w.hy+ey)
        self.values[:,self.blocked]=0

    def deposit(self,p: Array,channel: int,amount: float) -> None:
        if not self.enabled: return
        y,x=self.cell(p)
        if not self.blocked[y,x]: self.values[channel,y,x]=min(8.,float(self.values[channel,y,x])+amount)

    def spray(self,p: Array,amount: float = 4.,radius: float = 1.6) -> None:
        y,x=self.cell(p); rr=max(1,int(radius*self.width/(2*self.half[0])))
        for yy in range(max(0,y-rr),min(self.height,y+rr+1)):
            for xx in range(max(0,x-rr),min(self.width,x+rr+1)):
                if not self.blocked[yy,xx]: self.values[1,yy,xx]+=amount*np.exp(-((yy-y)**2+(xx-x)**2)/(rr*rr*.6))
        np.clip(self.values,0,8,out=self.values)

    def tick(self,dt: float) -> None:
        if not self.enabled: return
        coefficient=min(.18,.22*dt)
        free=~self.blocked; change=np.zeros_like(self.values)
        exchange=(self.values[:,:,1:]-self.values[:,:,:-1])*coefficient*(free[:,1:]&free[:,:-1])[None,:,:]
        change[:,:,:-1]+=exchange; change[:,:,1:]-=exchange
        exchange=(self.values[:,1:,:]-self.values[:,:-1,:])*coefficient*(free[1:,:]&free[:-1,:])[None,:,:]
        change[:,:-1,:]+=exchange; change[:,1:,:]-=exchange
        self.values+=change; self.values*=np.exp(-np.log(2.)*dt/np.asarray([45.,35.],np.float32)[:,None,None])
        self.values[:,self.blocked]=0; np.clip(self.values,0,8,out=self.values)

    def sample(self,p: Array,channel: int) -> float:
        if not self.enabled: return 0.
        y,x=self.cell(p); return float(self.values[channel,y,x])

    def sample_many(self, points: Array, channel: int) -> Array:
        if not self.enabled:
            return np.zeros(points.shape[:-1], np.float32)
        x = np.clip(((points[..., 0]+self.half[0])/(2*self.half[0])*self.width).astype(int), 0, self.width-1)
        y = np.clip(((points[..., 1]+self.half[1])/(2*self.half[1])*self.height).astype(int), 0, self.height-1)
        return self.values[channel, y, x]

    def gradient(self,p: Array,channel: int) -> Array:
        dx=np.asarray([.42,0],dtype=np.float32); dy=dx[::-1].copy()
        return np.asarray([self.sample(p+dx,channel)-self.sample(p-dx,channel),self.sample(p+dy,channel)-self.sample(p-dy,channel)],dtype=np.float32)
