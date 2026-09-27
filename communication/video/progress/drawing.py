"""Reusable v4 geometry; reads approved original artwork and toy paths."""
from __future__ import annotations
import math
import numpy as np
from manimlib import *
from communication.video.progress.plan import CHAPTERS
from communication.video.scenes.ant_drawing import Ant,Arena,BG,WHITE,BLUE,GOLD,TEAL,RED,GRAY,DIM,text,eq,ruler,line_curve
from communication.video.production.toy_motion import rollout,trail
from communication.video.stage.network import LivingNetwork

class ProgressScene(Scene):
    chapter=0
    def setup(self):
        super().setup();self.camera.background_color=BG;self.started=self.time
        self.duration=CHAPTERS[self.chapter].seconds
        self.scope=text('MECHANISM ILLUSTRATION',[-5.10,3.70,0],12,DIM)
        self.add(self.scope)
    @property
    def elapsed(self):return float(self.time-self.started)
    def at(self,t):
        if t-self.elapsed>.002:self.wait(t-self.elapsed)
    def finish(self):
        if self.elapsed>self.duration+.06:raise RuntimeError(f'{type(self).__name__}: overran {self.elapsed:.3f}/{self.duration}')
        self.at(self.duration)
    def scope_label(self,value):
        self.remove(self.scope);self.scope=text(value,[-4.95,3.70,0],12,DIM);self.add(self.scope)

class HeatMatrix(VGroup):
    """Small signed matrix retaining one square for each numeric element."""
    def __init__(self,values,center=ORIGIN,cell=.45):
        super().__init__();self.array=np.asarray(values,dtype=float);self.cells=VGroup()
        rows,cols=self.array.shape
        for i in range(rows):
            for j in range(cols):
                square=Square(side_length=cell-.045).set_stroke(GRAY,.7,.65)
                square.move_to([(j-(cols-1)/2)*cell,((rows-1)/2-i)*cell,0]);self.cells.add(square)
        self.add(self.cells);self.move_to(center);self.paint(values)
    def paint(self,values,scale=1.):
        for square,value in zip(self.cells,np.asarray(values).flat):
            v=float(value)/scale;square.set_fill(GOLD if v>=0 else BLUE,.045+.85*min(abs(v),1.))
        return self

def walking(scene,y=-1.45,speed=.43,size=.75,color=TEAL,start=-5.5):
    ts=np.linspace(0,scene.duration,401)
    points=np.column_stack([start+speed*ts,y+.11*np.sin(.7*ts),np.zeros_like(ts)])
    full=line_curve(points,color,2,.55);trace=full.copy().pointwise_become_partial(full,0,.001)
    ant=Ant(color,size)
    def update(mob):
        t=scene.elapsed;mob.pose_at([start+speed*t,y+.11*math.sin(.7*t),0],math.atan2(.077*math.cos(.7*t),speed),t*speed*13)
        trace.pointwise_become_partial(full,0,max(.001,min(1,t/scene.duration)))
    ant.add_updater(update)
    return ant,trace

def twins(scene,offset=0):
    top=Arena(rollout(22.,'fixed','shift'),[0,1.0,0],.94,.63,GRAY).follow(lambda:scene.elapsed+offset)
    bottom=Arena(rollout(22.,'always','shift'),[0,-1.6,0],.94,.63,TEAL).follow(lambda:scene.elapsed+offset)
    return top,bottom

def fit_formula(value,at,size=46,color=WHITE,width=12.1):
    result=eq(value,at,size,color)
    if result.get_width()>width:result.set_width(width)
    return result

def passing(edge,color=GOLD):
    return ShowPassingFlash(edge.copy().set_stroke(color,4),time_width=.4)
