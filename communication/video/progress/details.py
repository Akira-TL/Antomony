"""Local eligibility/modulation, bounded writes, evidence and deadline scenes."""
from __future__ import annotations
import math
import json
import numpy as np
from manimlib import *
from communication.video.progress.plan import ROOT
from communication.video.progress.drawing import (
 ProgressScene,HeatMatrix,walking,fit_formula,passing,LivingNetwork,
 Ant,Arena,BG,WHITE,BLUE,GOLD,TEAL,RED,GRAY,DIM,text,eq,ruler,line_curve,
)

class C04ModulatedChange(ProgressScene):
 chapter=4
 def construct(self):
  self.scope_label('LOCAL MECHANISM / SIGNED MODULATION')
  # Numbers here illustrate the exact outer-product form, not experiment results.
  x=np.array([.9,-.45,.60]);d=np.array([.75,-.40]);E=np.outer(x,d)
  left=HeatMatrix(E,[-3.4,1.35,0],.68)
  right=HeatMatrix(np.zeros((3,2)),[3.65,1.35,0],.68)
  llabel=eq('E_t',[-3.4,3.05,0],49,GOLD)
  rlabel=eq(r'\Delta F_t',[3.65,3.05,0],49,GOLD)
  ant,track=walking(self,y=-1.05,speed=.44,size=.66)
  self.add(ant,track,left,llabel)
  xs=VGroup(*(eq(f'{v:.2f}',[-4.70,2.03-i*.68,0],26,BLUE) for i,v in enumerate(x)))
  ds=VGroup(*(eq(f'{v:.2f}',[-3.74+j*.68,2.57,0],26,TEAL) for j,v in enumerate(d)))
  self.play(FadeIn(xs),FadeIn(ds),run_time=.7)
  self.at(2.1)
  for cell in left.cells:
   self.play(FlashAround(cell,color=GOLD),run_time=.24)
  axis=ruler([0,1.10,0],3.,-1,1)
  marker=Triangle().scale(.11).rotate(PI).set_color(TEAL)
  mu=ValueTracker(.70)
  number=DecimalNumber(.7,num_decimal_places=2,font_size=38).set_color(TEAL).move_to([0,2.6,0])
  mulabel=eq(r'\mu_t=',[-.9,2.6,0],40,TEAL)
  marker.add_updater(lambda m:m.move_to([1.5*mu.get_value(),1.38,0]))
  number.add_updater(lambda m:m.set_value(mu.get_value()))
  right.add_updater(lambda m:m.paint(mu.get_value()*E,scale=.70))
  self.at(4.7);self.play(ShowCreation(axis),FadeIn(marker),FadeIn(mulabel),FadeIn(number),FadeIn(right),FadeIn(rlabel),run_time=.8)
  # checkpoint: the-same-trace-can-produce-opposite-candidates
  self.at(8.8);self.play(mu.animate.set_value(1.),run_time=.8)
  self.play(mu.animate.set_value(0.),run_time=1.15)
  self.play(mu.animate.set_value(-.85),run_time=1.15)
  self.at(13.7)
  formula=eq(r'\Delta F_t=\eta\mu_t E_t',[0,-2.37,0],59)
  formula[r'\mu_t'].set_color(TEAL);formula['E_t'].set_color(GOLD)
  self.play(TransformFromCopy(llabel,formula),run_time=1.1)
  self.at(17.7)
  highlight=SurroundingRectangle(right,buff=.17).set_stroke(GOLD,1.3)
  self.play(ShowCreation(highlight),run_time=.6)
  self.at(20.5)
  right.clear_updaters();marker.clear_updaters();number.clear_updaters()
  self.play(FadeOut(right),FadeOut(rlabel),FadeOut(highlight),FadeOut(xs),FadeOut(ds),run_time=.45)
  center=np.array([3.65,1.15,0]);circle=Circle(radius=1.1).move_to(center).set_stroke(GRAY,1.7)
  raw=Arrow(center,center+np.array([1.50,.70,0]),buff=0,stroke_width=4).set_color(RED)
  bounded=Arrow(center,center+np.array([1.50,.70,0])/np.linalg.norm([1.50,.70])*1.1,buff=0,stroke_width=4).set_color(GOLD)
  label=text('bounded write',[3.65,2.7,0],28)
  self.play(ShowCreation(circle),GrowArrow(raw),FadeIn(label),run_time=.65)
  self.play(Transform(raw,bounded),run_time=.9)
  self.finish()

class C05FourSteps(ProgressScene):
 chapter=5
 def construct(self):
  self.scope_label('LOCAL MECHANISM / SIMULATION ACTIONS')
  ant,trace=walking(self,y=-.40,speed=.53,size=.78)
  self.add(trace,ant)
  pending=eq(r'\Delta F_t',[-3.9,2.65,0],54,GOLD)
  question=text('A proposal, not an order.',[1.15,2.75,0],32)
  self.play(Write(pending),FadeIn(question),run_time=.8)
  moments=[5.85+.675*i for i in range(16)]
  slots=VGroup();numbers=VGroup();events=VGroup()
  for i,moment in enumerate(moments):
   x=-5.8+11.6*i/15
   dot=Dot([x,1.40,0],radius=.055).set_color(BLUE)
   numeral=eq(str(i%4+1),[x,1.90,0],25,GRAY)
   def tick_update(m,dt,when=moment):m.set_opacity(.20 if self.elapsed<when else 1.)
   dot.add_updater(tick_update);numeral.add_updater(tick_update)
   slots.add(dot);numbers.add(numeral)
  baseline=Line([-6,1.4,0],[6,1.4,0]).set_stroke(DIM,1.3)
  self.at(3.8);self.play(FadeOut(question),ShowCreation(baseline),FadeIn(slots),FadeIn(numbers),run_time=.8)
  f=ValueTracker(0.)
  fname=eq('F=',[4.20,2.90,0],38,TEAL)
  fvalue=DecimalNumber(0,num_decimal_places=2,font_size=36).set_color(TEAL).move_to([5.2,2.90,0])
  fvalue.add_updater(lambda m:m.set_value(f.get_value()));self.add(fname,fvalue)
  # These apply/skip choices illustrate the interface, not learned performance.
  for index,decision in enumerate((1,0,0,1)):
   p=np.array([-5.8+11.6*(index*4+3)/15,1.4,0])
   ring=Circle(radius=.15).move_to(p).set_stroke(TEAL if decision else GRAY,2)
   ring.add_updater(lambda m,dt,k=index:m.set_opacity(float(self.elapsed>=moments[k*4+3])))
   events.add(ring)
  self.add(events)
  self.at(7.8);apply=text('APPLY',[-1.6,.60,0],27,TEAL)
  self.play(FadeIn(apply),f.animate.set_value(.35),run_time=.7)
  self.at(10.0);skip=text('SKIP',[1.45,.60,0],27,GRAY)
  self.play(FadeIn(skip),run_time=.6)
  memory=HeatMatrix(np.zeros((3,2)),[-4.0,-1.85,0],.33)
  def remember(m):
   t=self.elapsed;values=np.array([[math.sin(t),.4*math.cos(t*.7)],[math.sin(t*.6),math.cos(t*.4)],[math.cos(t*.5),math.sin(t*.9)]])
   m.paint(values)
  memory.add_updater(remember)
  memory_label=text('trace still updates',[-1.75,-1.85,0],25,GOLD)
  self.at(12.5);self.play(FadeIn(memory),FadeIn(memory_label),run_time=.65)
  formula=fit_formula(r'F_{t+1}=\Pi_{\mathcal B}\!\left(F_t+z_t\,\operatorname{clip}_{c}(\eta\mu_tE_t)\right)',[.6,-2.60,0],37,width=11.5)
  self.at(16.1);self.play(f.animate.set_value(.48),Write(formula),run_time=1.)
  self.finish()

class C06LearnTheRule(ProgressScene):
 chapter=6
 def construct(self):
  self.scope_label('TRAINING / DEPLOYMENT DISTINCTION')
  net=LivingNetwork(np.array([-2.65,.45,0]),.75)
  original_parameters=net.parameters
  net.parameters=lambda t:original_parameters(min(t,9.3)/7.44)
  net.add_updater(lambda m:m.update_at(self.elapsed))
  self.add(net)
  label=text('learning the update rule',[-2.65,2.45,0],29)
  self.play(FadeIn(label),run_time=.7)
  phi=DecimalNumber(.2,num_decimal_places=2,font_size=37).set_color(TEAL).move_to([-2.55,-1.2,0])
  def phi_update(m):
   t=min(self.elapsed,9.3);m.set_value(.2+.08*t+.1*math.sin(1.2*t))
  phi.add_updater(phi_update)
  self.add(eq(r'\phi=',[-3.5,-1.2,0],42,TEAL),phi)
  course=VGroup()
  for j in range(3):
   xx=np.linspace(1.3,5.6,80);yy=1.60-j*.60+.15*np.sin((xx-1.3)*(j+1)*2)
   curve=line_curve(np.column_stack((xx,yy,np.zeros_like(xx))),[BLUE,GOLD,TEAL][j],2,.70)
   course.add(curve)
  moving=VGroup(*(Dot(radius=.065).set_color(WHITE) for _ in range(3)))
  for j,dot in enumerate(moving):
   dot.add_updater(lambda m,dt,k=j:m.move_to(course[k].point_from_proportion((self.elapsed*.18+k*.22)%1)))
  self.at(3.6);self.play(ShowCreation(course),run_time=.8);self.add(moving)
  # checkpoint: outer-training-is-not-the-deployment-update
  self.at(6.8)
  reverse=net.edges.copy().set_stroke(RED,2.2,.7)
  for edge in reverse:edge.reverse_points()
  self.play(ShowPassingFlash(reverse,time_width=.38),run_time=1.2)
  ant,trace=walking(self,y=-2.2,speed=.58,size=.52)
  self.add(trace,ant)
  self.at(9.7)
  self.play(FadeOut(course),FadeOut(moving),Transform(label,text('using the learned rule',[-2.65,2.45,0],29)),run_time=.6)
  fast=HeatMatrix(np.zeros((3,2)),[3.6,.80,0],.58)
  def change(m):
   u=max(0,self.elapsed-10);m.paint(np.array([[.4*math.sin(u),.5*math.cos(u*.5)],[.3*math.sin(u*.7),-.4*math.sin(u*.4)],[.6*math.sin(u*.3),.2]]))
  fast.add_updater(change)
  self.play(FadeIn(fast),FadeIn(eq('F_t',[3.6,2.2,0],49,GOLD)),run_time=.8)
  self.at(14.0)
  fixed=text('rule parameters fixed',[-2.65,-.75,0],24,TEAL)
  local=text('fast connections adapt',[3.55,-.65,0],24,GOLD)
  self.play(FadeIn(fixed),FadeIn(local),run_time=.7)
  self.finish()

class C07SeparateTheClaims(ProgressScene):
 chapter=7
 def construct(self):
  self.scope_label('RECORDED EXPERIMENT / CONTINUOUS ADAPTATION')
  data=json.loads((ROOT/'communication/video/progress/provenance.json').read_text())['evidence']
  learned=float(data['mean_delivery_difference_learned_minus_skip'])
  always=learned-float(data['mean_delivery_difference_learned_minus_always'])
  self.add(text('Adaptation, or better update selection?',[0,3.05,0],34))
  unit=1.10;x0=-3.8
  axis=Line([x0,-1.95,0],[x0+7.4*unit,-1.95,0]).set_stroke(GRAY,1.3)
  ticks=VGroup()
  for n in (0,2,4,6):
   x=x0+n*unit;ticks.add(Line([x,-2.02,0],[x,-1.88,0]).set_stroke(GRAY,1.3),eq(str(n),[x,-2.25,0],25,GRAY))
  baseline=DashedLine([x0,-1.95,0],[x0,2.15,0],dash_length=.08).set_stroke(DIM,1.5)
  self.play(ShowCreation(axis),FadeIn(ticks),ShowCreation(baseline),run_time=.8)
  names=VGroup(text('No updates',[-5.3,1.55,0],24,GRAY),text('Learned',[-5.3,.25,0],24,TEAL),text('Always',[-5.3,-1.05,0],24,GOLD))
  self.add(names,Dot([x0,1.55,0],radius=.09).set_color(GRAY),eq('0',[x0+.35,1.55,0],29,GRAY))
  learned_line=Line([x0,.25,0],[x0+learned*unit,.25,0]).set_stroke(TEAL,4)
  learned_dot=Dot([x0+learned*unit,.25,0],radius=.11).set_color(TEAL)
  self.at(4.2);self.play(ShowCreation(learned_line),GrowFromCenter(learned_dot),run_time=1.1)
  self.play(FadeIn(eq('+5.25',learned_dot.get_center()+[.8,0,0],37,TEAL)),run_time=.5)
  always_line=Line([x0,-1.05,0],[x0+always*unit,-1.05,0]).set_stroke(GOLD,4)
  always_dot=Dot([x0+always*unit,-1.05,0],radius=.11).set_color(GOLD)
  self.at(8.2);self.play(ShowCreation(always_line),GrowFromCenter(always_dot),run_time=.9)
  self.play(FadeIn(eq('+6.25',always_dot.get_center()+[.72,0,0],37,GOLD)),run_time=.5)
  self.add(text('Mean extra deliveries vs. no updates',[.7,2.30,0],24,GRAY))
  note=text('4 world seeds · 3 changed conditions · one initialization set',[0,-2.62,0],17,GRAY)
  self.add(note)
  self.at(11.6)
  compare=eq(r'\Delta_{\rm learned-always}=-1.00',[.6,1.42,0],39)
  self.play(Write(compare),run_time=.8);self.finish()

class C08ActionDeadline(ProgressScene):
 chapter=8
 def construct(self):
  self.scope_label('ARCHITECTURE TARGET / NO MEASURED LATENCY CLAIM')
  ant,trace=walking(self,y=.35,speed=.77,size=.78)
  self.add(ant,trace)
  axis=Line([-5.9,-.50,0],[5.9,-.50,0]).set_stroke(DIM,1.7)
  dots=VGroup(*(Dot([x,-.50,0],radius=.045).set_color(BLUE) for x in np.linspace(-5.6,5.6,15)))
  pulse=Dot(radius=.08).set_color(WHITE)
  pulse.add_updater(lambda m:m.move_to([-5.6+min(1,self.elapsed/14)*11.2,-.50,0]))
  self.add(axis,dots,pulse)
  formula=fit_formula(r'T_{\rm sense\rightarrow act}\le D',[0,2.65,0],58)
  self.play(Write(formula),run_time=1.1)
  slots=VGroup(*(Square(side_length=.24).move_to([-3.6+i*.44,-1.8,0]).set_stroke(GRAY,1.0).set_fill(BG,1) for i in range(12)))
  work=text('preparing one local change',[0,-2.35,0],26,GOLD)
  def fill_work(m):
   n=int(np.clip((self.elapsed-3.8)/4.7,0,1)*12)
   for i,cell in enumerate(m):cell.set_fill(GOLD,.8 if i<n else .05)
  slots.add_updater(fill_work)
  self.at(3.8);self.play(FadeIn(slots),FadeIn(work),run_time=.7)
  active=text('last complete weights',[-2.8,1.55,0],27,BLUE)
  self.play(FadeIn(active),run_time=.6)
  self.at(8.5);slots.clear_updaters()
  commit=text('COMMIT',[3.65,1.55,0],29,TEAL)
  self.play(TransformFromCopy(slots,commit),FlashAround(ant,color=TEAL),run_time=.8)
  self.at(10.3);self.play(Transform(active,text('next action continues',[-2.8,1.55,0],27,TEAL)),FadeOut(work),run_time=.7)
  self.finish()

class C09KeepLearning(ProgressScene):
 chapter=9
 def construct(self):
  self.scope_label('ANT MODEL / ARCHITECTURE PROPOSAL')
  net=LivingNetwork(np.array([1.85,.15,0]),.72)
  net.add_updater(lambda m:m.update_at(self.elapsed+2))
  ant,trace=walking(self,y=-1.10,speed=.48,size=.85,start=-4.4)
  self.add(ant,trace)
  self.play(FadeIn(net),run_time=1.)
  title=text('Learning While Acting',[0,2.65,0],48)
  self.play(FadeIn(title),run_time=.8)
  self.at(2.1)
  edge=net.selected_edge();self.play(passing(edge),run_time=.8)
  ant.clear_updaters()
  self.play(ant.animate.scale(.25).move_to(net.layers[0][1]),run_time=.9)
  self.play(FadeOut(ant),FadeOut(trace),run_time=.45)
  self.add(text('Local feedback. Adaptive connections. Continuous action.',[0,-2.00,0],26),text('MathHackson',[0,-2.65,0],18,GRAY))
  self.finish()
