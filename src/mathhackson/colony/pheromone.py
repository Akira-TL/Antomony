"""两通道信息素；无通量墙界，有限差分不跨边界环绕。"""
from __future__ import annotations
import numpy as np
from .geometry import Array, Wall

class Pheromones:
    def __init__(self, width: int = 96, height: int = 64) -> None:
        self.width,self.height=width,height
        self.half=np.asarray([14.,10.],dtype=np.float32)
        self.values=np.zeros((2,height,width),dtype=np.float32)
        self.blocked=np.zeros((height,width),dtype=np.bool_)
        self.enabled=True

    def cell(self,p: Array) -> tuple[int,int]:
        x=int(np.clip((p[0]+14)/28*self.width,0,self.width-1))
        y=int(np.clip((p[1]+10)/20*self.height,0,self.height-1))
        return y,x

    def set_walls(self,walls: list[Wall]) -> None:
        x=np.linspace(-14+14/self.width,14-14/self.width,self.width)
        y=np.linspace(-10+10/self.height,10-10/self.height,self.height)
        self.blocked.fill(False)
        for w in walls:
            self.blocked |= (np.abs(x[None,:]-w.x)<=w.hx+14/self.width)&(np.abs(y[:,None]-w.y)<=w.hy+10/self.height)
        self.values[:,self.blocked]=0

    def deposit(self,p: Array,channel: int,amount: float) -> None:
        if not self.enabled: return
        y,x=self.cell(p)
        if not self.blocked[y,x]: self.values[channel,y,x]=min(8.,float(self.values[channel,y,x])+amount)

    def spray(self,p: Array,amount: float = 4.,radius: float = 1.6) -> None:
        y,x=self.cell(p); rr=max(1,int(radius*self.width/28))
        for yy in range(max(0,y-rr),min(self.height,y+rr+1)):
            for xx in range(max(0,x-rr),min(self.width,x+rr+1)):
                if not self.blocked[yy,xx]: self.values[1,yy,xx]+=amount*np.exp(-((yy-y)**2+(xx-x)**2)/(rr*rr*.6))
        np.clip(self.values,0,8,out=self.values)

    def tick(self,dt: float) -> None:
        if not self.enabled: return
        coefficient=min(.18,.7*dt)
        free=~self.blocked; change=np.zeros_like(self.values)
        exchange=(self.values[:,:,1:]-self.values[:,:,:-1])*coefficient*(free[:,1:]&free[:,:-1])[None,:,:]
        change[:,:,:-1]+=exchange; change[:,:,1:]-=exchange
        exchange=(self.values[:,1:,:]-self.values[:,:-1,:])*coefficient*(free[1:,:]&free[:-1,:])[None,:,:]
        change[:,:-1,:]+=exchange; change[:,1:,:]-=exchange
        self.values+=change; self.values*=np.exp(-np.log(2.)*dt/np.asarray([10.,6.],np.float32)[:,None,None])
        self.values[:,self.blocked]=0; np.clip(self.values,0,8,out=self.values)

    def sample(self,p: Array,channel: int) -> float:
        if not self.enabled: return 0.
        y,x=self.cell(p); return float(self.values[channel,y,x])

    def sample_many(self, points: Array, channel: int) -> Array:
        if not self.enabled:
            return np.zeros(points.shape[:-1], np.float32)
        x = np.clip(((points[..., 0]+14)/28*self.width).astype(int), 0, self.width-1)
        y = np.clip(((points[..., 1]+10)/20*self.height).astype(int), 0, self.height-1)
        return self.values[channel, y, x]

    def gradient(self,p: Array,channel: int) -> Array:
        dx=np.asarray([.42,0],dtype=np.float32); dy=dx[::-1].copy()
        return np.asarray([self.sample(p+dx,channel)-self.sample(p-dx,channel),self.sample(p+dy,channel)-self.sample(p-dy,channel)],dtype=np.float32)
