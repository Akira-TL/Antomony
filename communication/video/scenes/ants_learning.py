"""A moving experiment, rather than a procession of explanatory diagrams.

Each checkpoint can be replayed with ManimGL's interactive embed workflow.
The ant worlds are computed teaching examples, never research-result footage.
"""
from __future__ import annotations

import math
import numpy as np
from manimlib import *
from communication.video.scenes.ant_drawing import (
    Ant, Arena, AntFilm, BG, WHITE, BLUE, GOLD, TEAL, RED, GRAY, DIM,
    text, eq, ruler, line_curve, crumbs,
)
from communication.video.production.toy_motion import rollout, trail


def moving_dot_path(scene: AntFilm, color: str, y: float, speed: float = .40, size: float = .85) -> tuple[Ant, VMobject]:
    """A continuously walking ant used while a mathematical detail is enlarged."""
    points=np.array([[-5.5+speed*t,y+.13*np.sin(.9*t),0] for t in np.linspace(0,scene.duration,401)])
    template=line_curve(points,color,2,.55)
    trace=template.copy().pointwise_become_partial(template,0,.001)
    ant=Ant(color,size)
    def update(mob:Ant) -> None:
        t=scene.elapsed
        mob.pose_at([-5.5+speed*t,y+.13*np.sin(.9*t),0],math.atan2(.117*np.cos(.9*t),speed),t*speed*13)
        trace.pointwise_become_partial(template,0,max(.001,min(1,t/scene.duration)))
    ant.add_updater(update)
    return ant,trace


def steering_walk(scene: AntFilm, switch: float, before: float, after: float, push: float, y: float, speed: float, size: float, color: str = TEAL) -> tuple[Ant, VMobject]:
    """A parameter change changes commanded heading and subsequent trajectory."""
    def pose(t: float) -> tuple[np.ndarray, float, float]:
        first=min(t,switch);second=max(0.,t-switch)
        command=before if t<switch else after
        point=np.array([-5.5+speed*t,y+(before+push)*first+(after+push)*second,0.])
        phase=13*(math.hypot(speed,before)*first+math.hypot(speed,after)*second)
        return point,math.atan2(command,speed),phase
    points=np.array([pose(t)[0] for t in np.linspace(0,scene.duration,500)])
    template=line_curve(points,color,2,.55)
    trace=template.copy().pointwise_become_partial(template,0,.001)
    ant=Ant(color,size)
    def update(mob:Ant) -> None:
        mob.pose_at(*pose(scene.elapsed))
        trace.pointwise_become_partial(template,0,max(.001,min(1,scene.elapsed/scene.duration)))
    ant.add_updater(update)
    return ant,trace


def numeric(value: float, at: list[float], color: str = GOLD, size: float = 38) -> DecimalNumber:
    return DecimalNumber(value,num_decimal_places=2,font_size=size).set_color(color).move_to(at)


class A01FollowTheTrail(AntFilm):
    chapter=1
    def construct(self):
        world=Arena(rollout(18.,"fixed","shift"),ant_scale=.86).follow(lambda:self.elapsed)
        self.scope.fix_in_frame()
        self.frame.set_height(5.6).move_to([-2.65,.1,0])
        self.add(world)
        # checkpoint: walking-before-the-question
        self.play(self.frame.animate.set_height(8).move_to(ORIGIN),run_time=4.4,rate_func=smooth)
        note=text("a sideways push",[2.6,2.6,0],31,RED)
        self.at(5.0)
        self.play(FadeIn(note,.1*DOWN),run_time=.6)
        predicted=Ant(GRAY,.86).set_opacity(.22)
        gap=Line(ORIGIN,UP*.1).set_stroke(RED,2)
        def compare(mob:Ant):
            p=world.motion.pose(self.elapsed)
            mob.pose_at([p.x,float(trail(p.x)),0],0.,p.phase)
            a=np.array([p.x,float(trail(p.x)),0]);b=np.array([p.x,p.y,0])
            if np.linalg.norm(b-a)<.03:b=a+.03*UP
            gap.put_start_and_end_on(a,b)
        predicted.add_updater(compare)
        self.at(8.2)
        self.play(FadeIn(predicted),ShowCreation(gap),run_time=.65)
        q=text("Same action. Different outcome.",[0,-2.45,0],36)
        self.at(10.5)
        self.play(FadeIn(q),run_time=.7)
        self.at(14.0)
        self.play(FadeOut(note),FadeOut(predicted),FadeOut(gap),run_time=.6)
        self.finish()


class A02SameAntDifferentLearning(AntFilm):
    chapter=2
    def construct(self):
        top=Arena(rollout(20.,"fixed","shift"),[0,1.30,0],.93,.59,GRAY).follow(lambda:self.elapsed)
        bottom=Arena(rollout(20.,"always","shift"),[0,-1.6,0],.93,.59,TEAL).follow(lambda:self.elapsed)
        names=VGroup(text("fixed steering",[-4.5,3.02,0],29,GRAY),text("local correction",[-4.5,-.38,0],29,TEAL))
        self.add(top,bottom,names)
        same=text("same starting controller",[1.5,3.02,0],27)
        self.play(FadeIn(same),run_time=.6)
        self.at(4.5)
        self.play(FadeOut(same),run_time=.5)
        # checkpoint: paths-and-weights-change-together
        f0=eq("f=0",[4.1,3.0,0],42,GRAY)
        f1=eq("f=",[3.55,-.38,0],42,TEAL)
        value=numeric(0,[4.42,-.38,0],TEAL,37)
        value.add_updater(lambda m:m.set_value(bottom.motion.pose(self.elapsed).weight))
        self.at(6.)
        self.play(FadeIn(f0),FadeIn(f1),run_time=.7)
        self.add(value)
        underline=Line([-4.9,-3.13,0],[4.9,-3.13,0]).set_stroke(DIM,1)
        caption=text("Different update rules. Different paths.",[0,-3.43,0],28)
        self.at(14.0)
        self.play(ShowCreation(underline),FadeIn(caption),run_time=.8)
        self.finish()


class A03WeightBecomesMovement(AntFilm):
    chapter=3
    def construct(self):
        # This segment deliberately exposes a parameter and its physical effect.
        question=text("What changed?",[0,3.10,0],39)
        ant=Ant(TEAL,1.26)
        axis=ruler([.4,1.4,0],8.,-.5,1.5)
        sname=eq("s",[-1.6,2.1,0],47,BLUE)
        fname=eq("f",[1.1,2.1,0],47,GOLD)
        fixed=Dot([-1.6,1.4,0],radius=.09).set_color(BLUE)
        tip=Triangle().scale(.10).rotate(PI).set_color(GOLD)
        val=numeric(0,[4.5,2.05,0],GOLD,41)
        plus=eq("w=s+f",[3.25,-1.25,0],66)
        plus["s"].set_color(BLUE);plus["f"].set_color(GOLD)
        force=Arrow(ORIGIN,RIGHT,buff=0,stroke_width=5).set_color(WHITE)
        wind=Arrow(ORIGIN,UP,buff=0,stroke_width=3).set_color(RED)
        correction=Arrow(ORIGIN,DOWN,buff=0,stroke_width=4).set_color(GOLD)
        def weight(t:float)->float:
            if t<5.:return 0.
            if t<11.:return .85*smooth((t-5)/6)
            if t<15.:return .85
            if t<18.:return .85-.4*smooth((t-15)/3)
            if t<21.:return .45+.4*smooth((t-18)/3)
            return .85
        def drive(mob:Ant):
            t=self.elapsed;f=weight(t)
            p=np.array([-4.65+.13*t,-1.25+.1*math.sin(t*.6),0.])
            mob.pose_at(p,math.atan2(-f,.85),t*9)
            start=p+np.array([.85,.1,0])
            force.put_start_and_end_on(start,start+np.array([1.45,1.6*(.85-f),0]))
            wind.put_start_and_end_on(start+RIGHT*1.45,start+RIGHT*1.45+UP*1.36)
            correction.put_start_and_end_on(start,start+DOWN*max(.025,f*1.6))
            correction.set_opacity(1 if f>.01 else 0)
            tip.move_to([4*f-1.6,1.67,0])
            val.set_value(f)
        ant.add_updater(drive)
        ground=line_curve(np.array([[x,-2.35+.045*np.sin(x*3),0] for x in np.linspace(-6.4,.25,90)]),DIM,1.4,.7)
        self.add(ground,ant,force,wind)
        self.play(FadeIn(question),run_time=.7)
        self.at(2.9)
        self.play(FadeOut(question),ShowCreation(axis),FadeIn(fixed),FadeIn(tip),FadeIn(sname),FadeIn(fname),run_time=1.0)
        self.add(val)
        self.add(correction)
        self.at(7.0)
        keep=text("basic movement stays",[-3.55,-2.8,0],25,BLUE)
        self.play(FadeIn(keep),run_time=.6)
        self.at(11.4)
        self.play(Write(plus),run_time=1.6)
        answer=text("one small correction",[3.2,-2.23,0],30,GOLD)
        self.at(16.1)
        self.play(FadeIn(answer),run_time=.8)
        self.finish()


class A04LearnFromTheGap(AntFilm):
    chapter=4
    def construct(self):
        ant=Ant(BLUE,.90)
        ant.add_updater(lambda m:m.pose_at([-5.2+.19*self.elapsed,-1.15,0],0.,self.elapsed*6))
        track=Line([-6,-1.65,0],[-.4,-1.65,0]).set_stroke(TEAL,2,.35)
        self.add(ant,track)
        # checkpoint: a-geometric-error-becomes-an-equation
        origin=np.array([1.15,-.5,0])
        ax=VGroup(Line(origin,origin+RIGHT*4.5),Line(origin,origin+UP*3.2)).set_stroke(DIM,2)
        labels=VGroup(eq("x",origin+[4.75,0,0],39),eq("c",origin+[0,3.45,0],39))
        x=2.9;target=.85;w=ValueTracker(.20)
        pred=Dot(origin+[x,x*.2,0],radius=.10).set_color(BLUE)
        actual=Dot(origin+[x,x*target,0],radius=.10).set_color(WHITE)
        model=Line(origin,origin+[3.8,3.8*.2,0]).set_stroke(BLUE,3)
        error=Line(pred.get_center(),actual.get_center()).set_stroke(RED,5)
        cap=VGroup(Line(LEFT*.10,RIGHT*.10),Line(LEFT*.10,RIGHT*.10)).set_stroke(RED,2)
        pred_label=text("predicted",[5.25,.1,0],25,BLUE)
        actual_label=text("observed",[5.25,2.03,0],25)
        def update(mob:Line):
            z=w.get_value();loc=origin+np.array([x,x*z,0])
            mob.put_start_and_end_on(origin,origin+[3.8,3.8*z,0])
            pred.move_to(loc);error.put_start_and_end_on(loc,actual.get_center())
            cap[0].move_to(loc);cap[1].move_to(actual)
            pred_label.move_to(loc+[1.12,-.1,0])
        model.add_updater(update)
        formula=eq(r"\hat c=wx",[-3.5,1.85,0],62,BLUE)
        self.play(ShowCreation(ax),FadeIn(labels),ShowCreation(model),FadeIn(pred),Write(formula),run_time=1.2)
        self.play(FadeIn(pred_label),run_time=.4)
        self.at(4.0)
        self.play(FadeIn(actual,UP*.15),FadeIn(actual_label),ShowCreation(error),FadeIn(cap),run_time=.9)
        esym=eq(r"e=c-\hat c",[-3.5,.58,0],56,RED)
        self.at(6.)
        self.play(TransformFromCopy(error,esym),run_time=1.0)
        pieces=VGroup(eq(r"\Delta w",size=60,color=GOLD),eq("=",size=60),eq(r"\eta",size=60),eq("e",size=60,color=RED),eq("x",size=60,color=BLUE)).arrange(RIGHT,buff=.18).move_to([.1,-2.73,0])
        self.at(8.2)
        self.play(FadeIn(pieces[:3]),TransformFromCopy(esym,pieces[3]),TransformFromCopy(formula,pieces[4]),run_time=1.1)
        for value in (.5,.675,.77):
            self.play(w.animate.set_value(value),run_time=1.5,rate_func=smooth)
        local=text("only this local calculation",[-3.3,2.98,0],28,TEAL)
        self.play(FadeIn(local),run_time=.7)
        self.finish()


class A05NoiseOrChange(AntFilm):
    chapter=5
    def construct(self):
        policies=("fixed","always","selective")
        colors=(GRAY,GOLD,TEAL)
        centers=(1.75,-.24,-2.22)
        names=("fixed rule","update every time","selective updates")
        worlds=[]
        for policy,color,y,name in zip(policies,colors,centers,names):
            world=Arena(rollout(24.,policy,"mixed"),[0,y,0],.87,.47,color).follow(lambda:self.elapsed)
            worlds.append(world)
            self.add(world,text(name,[-4.68,y+.91,0],26,color))
            # A weight trace is drawn as the corresponding ant walks, not as a static chart.
            data=world.motion.samples
            pts=np.column_stack([-5.4+10.8*data[:,0]/24,y-.66+.20*data[:,5],np.zeros(len(data))])
            full=line_curve(pts,color,2,.8)
            progressive=full.copy().pointwise_become_partial(full,0,.001)
            progressive.add_updater(lambda mob,dt,curve=full:mob.pointwise_become_partial(curve,0,max(.001,min(1,self.elapsed/24))))
            self.add(Line([-5.5,y-.66,0],[5.5,y-.66,0]).set_stroke(DIM,1,.6),progressive,eq("f",[-5.9,y-.66,0],24,color))
        one=text("one brief gust",[2.5,3.15,0],34,RED)
        self.at(3.9)
        self.play(FadeIn(one),run_time=.5)
        self.at(7.0)
        self.play(FadeOut(one),run_time=.5)
        # Highlight a moment, without pretending the selective toy is a universal winner.
        question=text("The push is gone. Why keep correcting?",[0,3.25,0],29)
        self.play(FadeIn(question),run_time=.6)
        self.at(10.3)
        self.play(FadeOut(question),run_time=.5)
        lasting=text("a lasting change",[2.5,3.15,0],34,RED)
        self.at(12.0)
        self.play(FadeIn(lasting),run_time=.5)
        self.at(16.2)
        self.play(FadeOut(lasting),run_time=.5)
        # Compact equation occupies the empty upper band; paths stay on screen.
        rule=eq(r"F_{t+1}=F_t+z_t\Delta F_t",[1.1,3.21,0],43)
        rule["z_t"].set_color(TEAL);rule[r"\Delta F_t"].set_color(GOLD)
        self.play(Write(rule),run_time=1.1)
        self.finish()


class A06LearningTheChoice(AntFilm):
    chapter=6
    def construct(self):
        self.remove(self.scope)
        self.scope=text("PROPOSED LOCAL CREDIT",[-5.55,-3.72,0],12,DIM)
        self.add(self.scope)
        ant,path=moving_dot_path(self,TEAL,-1.80,.405,.78)
        self.add(path,ant)
        # No workflow boxes: a choice is tied to the place and time it happened.
        origin=np.array([-4.75,.15,0])
        axes=VGroup(Line(origin,origin+RIGHT*5.5),Line(origin,origin+UP*2.5)).set_stroke(DIM,1.8)
        labels=VGroup(text("evidence so far",[-2.0,-.27,0],26),text("update tendency",[-2.15,3.05,0],27,TEAL),eq("0",origin+LEFT*.25,22,GRAY),eq("1",origin+[-.25,2.3,0],22,GRAY))
        phi=ValueTracker(-2.1)
        xs=np.linspace(0,1,150)
        def curve_for(b:float)->VMobject:
            ys=1/(1+np.exp(-(4.8*xs+b)))
            return line_curve(origin+np.column_stack([xs*5.15,ys*2.2,np.zeros_like(xs)]),TEAL,3.5)
        old=curve_for(-2.1).set_stroke(GRAY,2,.4)
        curve=curve_for(-2.1)
        curve.add_updater(lambda m:m.set_points(curve_for(phi.get_value()).get_points()))
        self.play(ShowCreation(axes),FadeIn(labels),ShowCreation(old),ShowCreation(curve),run_time=1.3)
        pformula=eq(r"p_t=\sigma(\phi^\top\xi_t)",[3.65,2.6,0],43,TEAL)
        self.play(Write(pformula),run_time=.9)
        # checkpoint: a-trace-waits-for-real-consequences
        event_x=-3.3
        event=Dot([event_x,-1.80,0],radius=.075).set_color(GOLD)
        E=eq("E",[event_x,-2.78,0],45,GOLD)
        stalk=Line([event_x,-1.94,0],[event_x,-2.38,0]).set_stroke(GOLD,1.5)
        choice=text("this update",[event_x,-3.26,0],25,GOLD)
        self.at(4.7)
        self.play(FadeIn(event),ShowCreation(stalk),FadeIn(E),FadeIn(choice),run_time=.8)
        expect=eq(r"\widehat R",[3.05,.90,0],49,GRAY)
        observed=eq("R",[5.13,.90,0],49,WHITE)
        self.at(7.8)
        self.play(FadeIn(expect),run_time=.55)
        self.at(11.2)
        self.play(FadeIn(observed),run_time=.55)
        advantage=eq(r"A=R-\widehat R",[3.7,-.08,0],46)
        self.play(TransformFromCopy(VGroup(expect,observed),advantage),run_time=1.)
        local=eq(r"\Delta\phi=\beta A E",[2.5,-2.80,0],55,TEAL)
        self.at(15.0)
        self.play(TransformFromCopy(E,local),run_time=1.1)
        pulse=Dot([2.9,-2.5,0],radius=.075).set_color(TEAL)
        arc=ArcBetweenPoints(np.array([2.9,-2.5,0.]),np.array([-.1,1.32,0.]),angle=-.9)
        self.play(MoveAlongPath(pulse,arc),phi.animate.set_value(-.75),run_time=2.2)
        self.remove(pulse)
        self.at(21.0)
        mark=Dot(origin+[.58*5.15,2.2/(1+np.exp(-(4.8*.58-.75))),0],radius=.09).set_color(GOLD)
        self.play(FadeIn(mark),run_time=.55)
        update=text("the update rule changes too",[2.9,-3.38,0],27,TEAL)
        self.play(FadeIn(update),run_time=.7)
        self.finish()


class A07UndoOneChange(AntFilm):
    chapter=7
    def construct(self):
        ant,path=steering_walk(self,6.8,-.74375,-.38125,.60,-1.70,.68,.80)
        self.add(path,ant)
        ax=ruler([0,1.0,0],8.,0,1)
        self.play(ShowCreation(ax),run_time=.7)
        left=np.array([-4,1.0,0]);first=Arrow(left,left+RIGHT*2.9,buff=0,stroke_width=8).set_color(GOLD)
        second=Arrow(left+RIGHT*2.9,left+RIGHT*5.95,buff=0,stroke_width=8).set_color(TEAL)
        d1=eq(r"\delta_j",[-2.55,1.8,0],52,GOLD)
        d2=eq(r"\delta_k",[.48,1.8,0],52,TEAL)
        total=Dot(left+RIGHT*5.95,radius=.09).set_color(WHITE)
        self.play(GrowArrow(first),GrowArrow(second),FadeIn(d1),FadeIn(d2),FadeIn(total),run_time=1.3)
        reduction=eq(r"F\leftarrow F-r\delta_j",[2.65,2.7,0],51)
        reduction[r"\delta_j"].set_color(GOLD)
        self.at(3.8)
        self.play(Write(reduction),run_time=1.0)
        self.play(FadeOut(first,UP*.6),FadeOut(d1,UP*.6),second.animate.shift(LEFT*2.9),d2.animate.shift(LEFT*2.9),total.animate.shift(LEFT*2.9),run_time=2.)
        # Earlier footprints remain fixed while the ant continues forward.
        footprints=VGroup(*(Dot([-5.2+i*.38,-2.4,0],radius=.022).set_color(GRAY) for i in range(15)))
        self.play(FadeIn(footprints),run_time=.6)
        remembered=text("the path already walked",[-2.5,-2.97,0],28,GRAY)
        self.at(8.8)
        self.play(FadeIn(remembered),run_time=.6)
        self.finish()


class A08KeepWalking(AntFilm):
    chapter=8
    def construct(self):
        self.remove(self.scope)
        self.scope=text("REAL-TIME DESIGN TARGET",[-5.4,-3.72,0],12,DIM)
        self.add(self.scope)
        ant,path=steering_walk(self,12.4,0.,-.30,.13,-1.60,.55,.93,BLUE)
        self.add(path,ant)
        # Every moving footfall corresponds to a continuing action, not a stalled box.
        footmarks=VGroup(*(Dot([-5.25+i*1.38,-2.25,0],radius=.05).set_color(DIM) for i in range(9)))
        self.add(footmarks)
        self.at(1.8)
        name=text("next action",[-4.65,-2.92,0],27,BLUE)
        self.play(FadeIn(name),run_time=.6)
        for i,dot in enumerate(footmarks):
            dot.add_updater(lambda m,dt,j=i:m.set_color(BLUE if self.elapsed>(j+1)*1.8 else DIM))
        # A genuine little outer product fills gradually while the ant is walking.
        entries=[]
        matrix=VGroup()
        e=np.array([.4,-.2,.1]);x=np.array([1.,.5,-.5])
        for r in range(3):
            for c in range(3):
                p=np.array([-1.9+c*.80,2.45-r*.62,0])
                cell=Square(.62).set_stroke(DIM,1).set_fill(GOLD,.05).move_to(p)
                val=eq(f"{e[r]*x[c]:+.2f}",p,27,GOLD).set_opacity(0)
                entries.append(val);matrix.add(cell,val)
        product=eq(r"\Delta F=\eta e x^\top",[2.35,2.05,0],50,GOLD)
        self.at(4.0)
        self.play(FadeIn(matrix),Write(product),run_time=1.0)
        for i,mob in enumerate(entries):
            self.play(mob.animate.set_opacity(1),run_time=.40)
        pending=text("not ready yet",[2.3,1.02,0],27,GOLD)
        self.play(FadeIn(pending),run_time=.55)
        self.at(11.2)
        self.play(FadeOut(pending),matrix.animate.set_color(TEAL),run_time=.6)
        commit=text("commit",[2.8,.6,0],30,TEAL)
        self.play(FadeIn(commit),run_time=.6)
        committed=eq("f",[2.8,.12,0],37,TEAL)
        self.play(TransformFromCopy(matrix,committed),run_time=.7)
        deadline=eq(r"T_{\mathrm{sense}\to\mathrm{act}}\leq D",[1.5,3.3,0],44)
        self.at(14.3)
        self.play(Write(deadline),run_time=1.1)
        self.finish()


class A09ReturnToTheAnts(AntFilm):
    chapter=9
    def construct(self):
        for y,policy,color,name in ((1.7,"fixed",GRAY,"fixed rule"),(-.2,"always",GOLD,"always update"),(-2.1,"selective",TEAL,"selective updates")):
            world=Arena(rollout(24.,policy,"mixed"),[0,y,0],.86,.44,color).follow(lambda:self.elapsed+10)
            self.add(world,text(name,[-4.8,y+.9,0],25,color))
        self.at(3.)
        title=text("Learning While Acting",[1.45,3.25,0],43)
        self.play(FadeIn(title),run_time=.9)
        self.at(8.5)
        end=text("Can it learn how to learn?",[0,-3.35,0],33)
        self.play(FadeIn(end),run_time=.8)
        self.finish()
