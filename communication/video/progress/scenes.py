"""Original v4 mathematical animation: causal activity, local changes, moving ants.

English artwork only. Subtitles are produced separately by voice.py.
Checkpoint comments support the existing ManimGL interactive editing workflow.
"""
from __future__ import annotations
import math
import numpy as np
from manimlib import *
from communication.video.progress.drawing import (
 ProgressScene,HeatMatrix,walking,twins,fit_formula,passing,LivingNetwork,
 Ant,Arena,BG,WHITE,BLUE,GOLD,TEAL,RED,GRAY,DIM,text,eq,ruler,line_curve,rollout,trail,
)
from communication.video.progress import details as detail_scenes

# ManimGL discovers classes defined in this scene module, not imported classes.
class C04ModulatedChange(detail_scenes.C04ModulatedChange):
 pass
class C05FourSteps(detail_scenes.C05FourSteps):
 pass
class C06LearnTheRule(detail_scenes.C06LearnTheRule):
 pass
class C07SeparateTheClaims(detail_scenes.C07SeparateTheClaims):
 pass
class C08ActionDeadline(detail_scenes.C08ActionDeadline):
 pass
class C09KeepLearning(detail_scenes.C09KeepLearning):
 pass

class C00LearningContinues(ProgressScene):
 chapter=0
 def construct(self):
  self.scope_label('TRAIN-THEN-DEPLOY EXAMPLE')
  net=LivingNetwork(np.array([-2.6,.15,0]),.90)
  net.add_updater(lambda n:n.update_at(self.elapsed));self.add(net)
  question=text('Training finished.',[-2.6,2.95,0],38)
  self.play(FadeIn(question),run_time=.65)
  dials=VGroup();needles=VGroup()
  for i,x in enumerate((-4.3,-2.7,-1.1)):
   point=np.array([x,2.0,0]);circle=Circle(radius=.22).move_to(point).set_stroke(GRAY,1.2)
   needle=Line(point,point+.19*RIGHT).set_stroke(GOLD,2.8)
   def needle_update(m,dt,p=point,index=i):
    v=float(net.parameters(self.elapsed)[0][index,0]);a=.6-v
    m.put_start_and_end_on(p,p+.19*np.array([math.cos(a),math.sin(a),0]))
   needle.add_updater(needle_update);dials.add(circle);needles.add(needle)
  self.add(dials,needles)
  self.at(2.2);self.play(Transform(question,text('Learning finished?',[-2.6,2.95,0],38,GOLD)),run_time=.7)
  # checkpoint: fixed-weights-moving-activity
  axis=VGroup(Line([1.45,-.85,0],[6.25,-.85,0]),Line([1.45,-.85,0],[1.45,1.8,0])).set_stroke(DIM,1.4)
  legends=VGroup(text('prediction',[2.4,2.25,0],24,BLUE),text('observation',[4.9,2.25,0],24,WHITE))
  cache_t=np.linspace(0,25,2001);cache_y=np.array([net.prediction(float(t)) for t in cache_t])
  pred=line_curve(np.array([[1.5,0,0],[1.51,0,0]]),BLUE,3)
  obs=pred.copy().set_stroke(WHITE,2.3);dot=Dot(radius=.07).set_color(WHITE)
  gap=Line(ORIGIN,UP*.02).set_stroke(RED,4);graph=VGroup(pred,obs,dot,gap)
  def graph_update(m):
   end=max(.1,self.elapsed-.30);start=max(0,end-5.4);times=np.linspace(start,end,70)
   xx=1.6+4.45*(times-start)/5.4;yy=np.interp(times,cache_t,cache_y)*1.2+.25
   actual=yy+.70*(times>=6.4)
   pred.set_points_as_corners(np.column_stack((xx,yy,np.zeros_like(xx))))
   obs.set_points_as_corners(np.column_stack((xx,actual,np.zeros_like(xx))))
   dot.move_to([xx[-1],actual[-1],0]);gap.put_start_and_end_on(np.array([xx[-1],yy[-1],0]),np.array([xx[-1],actual[-1]+.001,0]));gap.set_opacity(float(end>=6.4))
  graph.add_updater(graph_update)
  formula=fit_formula(r'\hat y_t=f_\theta(x_t)',[3.85,-1.65,0],49)
  self.at(4.8);self.play(FadeIn(axis),FadeIn(legends),Write(formula),run_time=.7);self.add(graph)
  self.at(8.0);self.play(FlashAround(dot,color=RED),run_time=.7)
  edge=net.selected_edge();local=Ellipse(width=2.6,height=2.5).move_to(edge.get_center()).set_stroke(GOLD,1.5)
  candidate=DashedLine(edge.get_start()+UP*.13,edge.get_end()+UP*.13,dash_length=.07).set_stroke(GOLD,3)
  delta=eq(r'\Delta\theta_{\rm local}',[-1.45,-1.80,0],44,GOLD)
  self.at(10.2);self.play(FadeOut(question),FadeOut(dials),FadeOut(needles),ShowCreation(local),ShowCreation(candidate),Write(delta),run_time=1.1)
  self.at(14.7);self.play(passing(edge),run_time=.9)
  short=text('A small change. While the world keeps moving.',[0,-2.62,0],28)
  self.play(FadeIn(short),run_time=.6)
  # checkpoint: the-question-enters-a-moving-world
  self.at(19.0);graph.clear_updaters();net.clear_updaters()
  world=Arena(rollout(22.,'fixed','shift'),[0,1.0,0],.94,.63,GRAY)
  pos=world.ant.get_center()
  self.play(net.animate.scale(.13).move_to(pos),FadeOut(axis),FadeOut(legends),FadeOut(graph),FadeOut(formula),FadeOut(local),FadeOut(candidate),FadeOut(delta),FadeOut(short),run_time=1.6)
  self.play(FadeIn(world),FadeOut(net),run_time=.9)
  world.follow(lambda:max(0.,self.elapsed-22.));self.scope_label('TOY ILLUSTRATION')
  self.finish()

class C01MatchedAnts(ProgressScene):
 chapter=1
 def construct(self):
  self.scope_label('TOY ILLUSTRATION')
  top,bottom=twins(self,2.)
  self.add(top);self.play(FadeIn(bottom),run_time=.6)
  names=VGroup(text('fixed weights',[-4.85,2.85,0],28,GRAY),text('local correction',[-4.6,-.10,0],28,TEAL))
  self.play(FadeIn(names),run_time=.6)
  same=text('Same start. Same disturbance.',[2.2,3.0,0],27)
  self.add(same)
  self.at(4.);self.play(Indicate(top.wind,color=RED),Indicate(bottom.wind,color=RED),run_time=.8)
  fixed=eq('f=0',[4.55,2.8,0],43,GRAY)
  label=eq('f=',[3.8,-.12,0],43,TEAL)
  number=DecimalNumber(0,num_decimal_places=2,font_size=37).set_color(TEAL).move_to([4.65,-.12,0])
  number.add_updater(lambda m:m.set_value(bottom.motion.pose(self.elapsed+2.).weight))
  self.at(7.6);self.play(FadeOut(same),FadeIn(fixed),run_time=.7)
  self.at(10.4);self.play(FadeIn(label),run_time=.5);self.add(number)
  self.at(14.7)
  top_trace=top.track.copy().set_stroke(GRAY,4);bottom_trace=bottom.track.copy().set_stroke(TEAL,4)
  self.play(ShowPassingFlash(top_trace,time_width=.25),ShowPassingFlash(bottom_trace,time_width=.25),run_time=1.1)
  self.at(17.1);q=text('What changed inside?',[0,-2.72,0],32,GOLD);self.play(FadeIn(q),run_time=.6)
  self.finish()

class C02StableAndFast(ProgressScene):
 chapter=2
 def construct(self):
  self.scope_label('TOY ILLUSTRATION / PARAMETER VIEW')
  ant=Ant(TEAL,1.13);axis=ruler([0,1.2,0],8,-.5,1.5)
  marker=Triangle().scale(.105).rotate(PI).set_color(GOLD)
  value=DecimalNumber(0,num_decimal_places=2,font_size=38).set_color(GOLD).move_to([4.7,2.0,0])
  active=Arrow(ORIGIN,RIGHT,buff=0,stroke_width=4).set_color(WHITE)
  counter=Arrow(ORIGIN,DOWN,buff=0,stroke_width=4).set_color(GOLD)
  def weight(t):return .85*smooth(float(np.clip((t-5.8)/3.7,0,1)))
  def drive(m):
   t=self.elapsed;f=weight(t);p=np.array([-4.9+.16*t,-1.0+.05*math.sin(t),0])
   m.pose_at(p,math.atan2(-f,.85),t*5.5)
   start=p+[.8,.08,0];active.put_start_and_end_on(start,start+[1.5,1.4*(.85-f),0])
   counter.put_start_and_end_on(start,start+DOWN*max(.01,f*1.4));counter.set_opacity(float(f>.01))
   marker.move_to([4*f-2.,1.45,0]);value.set_value(f)
  ant.add_updater(drive);self.add(ant,active,counter)
  self.add(line_curve(np.array([[x,-2.,0] for x in np.linspace(-6.2,-.3,30)]),DIM,1.5))
  self.at(1.8);self.play(ShowCreation(axis),FadeIn(marker),run_time=.8);self.add(value)
  flabel=eq('f',[3.9,2.,0],42,GOLD);self.add(flabel)
  base_values=np.array([[.4,-.2],[.1,.7],[-.3,.2]])
  stable=HeatMatrix(base_values,[1.7,-.7,0],.47)
  fast=HeatMatrix(np.zeros((3,2)),[3.05,-.35,0],.47)
  fast.add_updater(lambda m:m.paint(np.array([[0,weight(self.elapsed)],[0,0],[0,0]])))
  slabel=eq('S',[1.7,.35,0],35,BLUE);flabel2=eq('F_t',[3.05,.72,0],35,GOLD)
  self.at(3.3);self.play(FadeIn(stable),FadeIn(fast),FadeIn(slabel),FadeIn(flabel2),run_time=.8)
  self.at(9.9);fast.clear_updaters()
  merged=HeatMatrix(base_values+np.array([[0,.85],[0,0],[0,0]]),[3.5,-.8,0],.56)
  self.play(TransformFromCopy(stable,merged),fast.animate.move_to(merged),FadeOut(slabel),FadeOut(flabel2),run_time=1.1)
  self.remove(fast);self.play(FadeOut(stable),run_time=.4)
  formula=eq('W_t=S+F_t',[0,2.75,0],60);formula['S'].set_color(BLUE);formula['F_t'].set_color(GOLD)
  self.play(Write(formula),run_time=1.2)
  note=text('Same motor skills. A different correction.',[1.2,-2.45,0],26)
  self.play(FadeIn(note),run_time=.6);self.finish()

class C03Eligibility(ProgressScene):
 chapter=3
 def construct(self):
  self.scope_label('LOCAL MECHANISM / LOW-DIMENSIONAL VIEW')
  ant,track=walking(self,y=-1.15,speed=.44,size=.76);self.add(track,ant)
  event_times=(3.,7.,11.,15.,19.)
  features=np.array([[.8,.3,-.4],[.2,.9,.5],[-.1,.7,.6],[.5,-.4,.2],[.3,.2,.9]])
  deviations=np.array([[.3,.7],[-.6,.25],[.2,-.4],[.6,.1],[-.2,.6]])
  records=VGroup();bars=VGroup()
  for i,t in enumerate(event_times):
   x=-5.5+.44*t
   dot=Dot([x,-.85,0],radius=.075).set_color(GOLD).set_opacity(0)
   bar=Line([x,.05,0],[x,.07,0]).set_stroke(GOLD,5,0)
   def fade_record(m,dt,index=i,stamp=t,pos=x):
    age=self.elapsed-stamp
    if age<0:m.set_opacity(0);bars[index].set_opacity(0)
    else:
     strength=.93**int(age);m.set_opacity(.12+.78*strength)
     bars[index].put_start_and_end_on(np.array([pos,.05,0]),np.array([pos,.05+1.65*strength,0]));bars[index].set_opacity(.2+.7*strength)
   bars.add(bar);records.add(dot);dot.add_updater(fade_record)
  self.add(records,bars)
  early=text('action now',[-4.4,-.36,0],25,GOLD)
  late=text('consequence later',[3.4,2.75,0],31,WHITE)
  self.play(FadeIn(early),run_time=.7)
  self.at(4.4);question=text('Which earlier activity matters?',[0,2.8,0],34)
  self.play(FadeIn(question),run_time=.7)
  self.at(7.8);self.play(FadeOut(early),FadeOut(question),run_time=.6)
  title=text('a fading local trace',[-2.3,2.45,0],29,GOLD)
  self.play(FadeIn(title),run_time=.6)
  memory=HeatMatrix(np.zeros((3,2)),[4.5,.90,0],.55)
  def update_matrix(m):
   E=np.zeros((3,2))
   for t,x,d in zip(event_times,features,deviations):
    if self.elapsed>=t:E+=(1-.93)*.93**int(self.elapsed-t)*np.outer(x,d)
   m.paint(E,scale=.07)
  memory.add_updater(update_matrix)
  self.at(12.7);self.play(FadeIn(memory),run_time=.7)
  symbol=eq('E_t',[4.5,2.05,0],44,GOLD);self.play(FadeIn(symbol),run_time=.5)
  # checkpoint: a-delayed-outcome-finds-the-still-present-trace
  self.at(16.2);self.play(Transform(title,text('Eligibility trace',[-2.3,2.45,0],32,GOLD)),FadeIn(late),run_time=.8)
  feedback=Dot([5.1,2.55,0],radius=.12).set_color(WHITE)
  arc=ArcBetweenPoints(feedback.get_center(),[4.5,1.65,0],angle=-PI/2).set_stroke(WHITE,1.7)
  self.add(feedback);self.play(MoveAlongPath(feedback,arc),FlashAround(memory,color=GOLD),run_time=1.)
  formula=fit_formula(r'E_t=\lambda E_{t-1}+(1-\lambda)x_t^{\mathsf T}(d_t-\bar d_t)',[0,-2.45,0],44)
  self.at(19.3);self.play(Write(formula),run_time=1.4)
  self.finish()
