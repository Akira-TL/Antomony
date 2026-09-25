"""每个体独立两隐层预测网络；学习已执行动作后果，而非读取全局路线。"""
from __future__ import annotations
import hashlib
import numpy as np
from numpy.typing import NDArray
from mathhackson.fast_residual import FastResidualParameter
Array=NDArray[np.float32]

class Brain:
    input_width=8
    def __init__(self,seed: int,warmup: int=220) -> None:
        self.rng=np.random.default_rng(seed)
        self.w1=self.rng.normal(0,.35,(8,24)).astype(np.float32); self.b1=np.zeros(24,np.float32)
        self.w2=self.rng.normal(0,.22,(24,16)).astype(np.float32); self.b2=np.zeros(16,np.float32)
        w=self.rng.normal(0,.15,(17,3)).astype(np.float32)
        self.head=FastResidualParameter(w,recent_capacity=8)
        self.updates=0; self.last_loss=0.; self.last_delta=0.; self.frozen=False
        self.hidden=np.zeros(16,np.float32); self.output=np.zeros(3,np.float32)
        # 各自生成局部练习；监督标签是动作造成的位移/接触，不是正确路线。
        x,y=self.practice(768); hold_x,hold_y=self.practice(128)
        self.birth_loss=float(np.mean((self.predict(hold_x)-hold_y)**2))
        for _ in range(warmup):
            indices=self.rng.integers(0,len(x),48)
            self.train(x[indices],y[indices],rate=.12,all_layers=True,record=False)
        self.warm_loss=float(np.mean((self.predict(hold_x)-hold_y)**2))
        self.head=FastResidualParameter(self.head.effective,recent_capacity=8)
        self.birth_head=self.head.effective.copy(); self.rng=np.random.default_rng(seed+700001)

    def practice(self,n: int) -> tuple[Array,Array]:
        angle=self.rng.uniform(-np.pi,np.pi,n); clear=self.rng.uniform(0,1,n)
        clear[:n//3]*=.10
        x=self.rng.uniform(0,1,(n,8)).astype(np.float32)
        x[:,0]=np.cos(angle); x[:,1]=np.sin(angle); x[:,2]=clear
        x[:,5]=self.rng.integers(0,2,n); x[:,6:8]=0
        mobility=np.clip(clear*2.4/.18,0,1)
        y=np.column_stack((np.cos(angle)*mobility,np.sin(angle)*mobility,(mobility<.999).astype(float))).astype(np.float32)
        return x,y

    def features(self,x: Array) -> tuple[Array,Array,Array]:
        a=np.tanh(x@self.w1+self.b1); b=np.tanh(a@self.w2+self.b2)
        phi=np.concatenate((b,np.ones((len(b),1),np.float32)),axis=1)
        return a,b,phi

    def predict(self,x: Array) -> Array:
        _,_,phi=self.features(np.atleast_2d(x)); return phi@self.head.effective

    def inspect(self,x: Array) -> Array:
        _,b,phi=self.features(np.atleast_2d(x)); self.hidden=b[0].copy()
        self.output=(phi@self.head.effective)[0].copy(); return self.output.copy()

    def train(self,x: Array,y: Array,*,rate: float=.045,all_layers: bool=False,record: bool=True) -> float:
        x=np.atleast_2d(x); y=np.atleast_2d(y)
        a,b,phi=self.features(x); old=self.head.effective; error=phi@old-y
        loss=float(np.mean(error*error))
        if self.frozen: return loss
        derivative=np.clip(error,-2,2)/len(x)
        delta=-rate*(phi.T@derivative)/(1+float(np.mean(np.sum(phi*phi,axis=1)))*.15)
        if all_layers:
            db=(derivative@old[:-1].T)*(1-b*b)
            da=(db@self.w2.T)*(1-a*a)
            self.w2-=rate*(a.T@db); self.b2-=rate*db.sum(0)
            self.w1-=rate*(x.T@da); self.b1-=rate*da.sum(0)
        self.head.add_delta(delta)
        if record:
            self.updates+=1; self.last_loss=loss; self.last_delta=float(np.linalg.norm(delta))
        return loss

    def fingerprint(self) -> str:
        return hashlib.sha256(self.head.effective.tobytes()).hexdigest()[:12]

    @property
    def drift(self) -> float:
        return float(np.linalg.norm(self.head.effective-self.birth_head))
