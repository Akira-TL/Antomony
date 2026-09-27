"""Articulated vector ants and continuously evaluated teaching-world drawings."""
from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import json

import numpy as np
from manimlib import (
    Scene, VGroup, VMobject, Mobject, Circle, Ellipse, Line, Arrow, Polygon,
    Dot, Text, Tex, DecimalNumber, FadeIn, FadeOut, ORIGIN, UP, RIGHT, LEFT, DOWN,
)
from manimlib.config import manim_config
from communication.video.film_plan import DURATIONS
from communication.video.production.toy_motion import Motion, trail

manim_config.tex.template = "basic"
BG = "#0B0E16"
WHITE = "#F2EEE6"
BLUE = "#70B8E8"
GOLD = "#FFD166"
TEAL = "#57D6BD"
RED = "#F4777E"
GRAY = "#A4A9B5"
DIM = "#455363"
FONT = "Inter"
ROOT = Path(__file__).resolve().parents[3]
Point = np.ndarray | list[float] | tuple[float, float, float]


def text(value: str, at: Point = ORIGIN, size: float = 32, color: str = WHITE) -> Text:
    return Text(value, font=FONT, font_size=size).set_color(color).move_to(at)


def eq(value: str, at: Point = ORIGIN, size: float = 54, color: str = WHITE) -> Tex:
    return Tex(value, font_size=size).set_color(color).move_to(at)


def line_curve(points: np.ndarray, color: str, width: float = 2, opacity: float = 1) -> VMobject:
    return VMobject().set_points_as_corners(points).set_stroke(color, width, opacity).set_fill(opacity=0)


class Ant(VGroup):
    """Six independently articulated legs, head, waist, abdomen and antennae.

    Pose updates transform cached geometry rather than allocating new sprites.
    Phase is supplied by distance travelled; a stationary ant needn't march.
    """
    def __init__(self, color: str = BLUE, scale: float = 1.0):
        super().__init__()
        self.tint = color
        self.unit = scale
        self.legs = VGroup(*(VMobject().set_stroke(color, 2.7) for _ in range(6)))
        self.knees = VGroup(*(Circle(radius=.023).set_fill(color,1).set_stroke(WHITE,.6,.4) for _ in range(6)))
        abdomen = Ellipse(width=0.64, height=0.43).move_to([-0.49, 0, 0])
        waist = Ellipse(width=0.15, height=0.13).move_to([-0.13, 0, 0])
        thorax = Ellipse(width=0.40, height=0.27).move_to([0.10, 0, 0])
        neck = Ellipse(width=0.12, height=0.17).move_to([0.31, 0, 0])
        head = Ellipse(width=0.36, height=0.34).move_to([0.48, 0, 0])
        self.shell = VGroup(abdomen, waist, thorax, neck, head)
        self.shell.set_fill(color, 0.85).set_stroke(WHITE, 0.8, 0.4)
        self.shade = VGroup(*(
            Ellipse(width=w, height=h).move_to([x, -0.07, 0]).set_fill(BG, 0.45).set_stroke(width=0)
            for x, w, h in [(-0.52, 0.46, 0.22), (0.10, 0.27, 0.13), (0.47, 0.24, 0.15)]
        ))
        self.shine = VGroup(*(
            Ellipse(width=w, height=h).move_to([x, 0.095, 0]).set_fill(WHITE, 0.30).set_stroke(width=0)
            for x, w, h in [(-0.48, 0.37, 0.048), (0.09, 0.21, 0.035), (0.46, 0.17, 0.035)]
        ))
        self.eyes = VGroup(*(
            Circle(radius=0.039).move_to([0.54, side * 0.115, 0]).set_fill(BG, 1).set_stroke(WHITE, 0.5)
            for side in (-1, 1)
        ))
        self.mandibles = VGroup(*(
            VMobject().set_points_as_corners([[0.64, .07*s, 0], [.75,.06*s,0], [.74,.025*s,0]]).set_stroke(color, 1.9)
            for s in (-1,1)
        ))
        self.antennae = VGroup(*(VMobject().set_stroke(color, 1.8) for _ in range(2)))
        self.shadow = Ellipse(width=1.38, height=0.82).set_fill("#000000", 0.24).set_stroke(width=0)
        self.add(self.shadow, self.legs, self.knees, self.shell, self.shade, self.shine, self.eyes, self.mandibles, self.antennae)
        self.rigid = [m for group in (self.shell, self.shade, self.shine, self.eyes, self.mandibles) for m in group]
        self.base_points = tuple(m.get_points().copy() for m in self.rigid)
        self.shadow_base = self.shadow.get_points().copy()
        self.pose_at(ORIGIN, 0.0, 0.0)

    def pose_at(self, center: Point, heading: float, phase: float, unit: float | None = None) -> "Ant":
        scale = self.unit if unit is None else unit
        center = np.asarray(center, dtype=float)
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        bob = np.array([0, 0.012 * math.sin(2*phase), 0])
        for mob, original in zip(self.rigid, self.base_points):
            mob.set_points((original + bob) @ rotation * scale + center)
        self.shadow.set_points(self.shadow_base @ rotation * scale + center + [0,-.055*scale,0])
        for side_idx, side in enumerate((-1,1)):
            for j, base_x in enumerate((-.08,.08,.24)):
                angle = phase + (j + side_idx) * math.pi
                swing = math.sin(angle)
                lift = max(0., math.cos(angle))
                anchor = np.array([base_x, .10*side, 0.])
                knee = np.array([base_x + (j-1)*.18 + .07*swing, (.34-.018*lift)*side, 0.])
                foot = np.array([base_x - .26 + (j-1)*.26 + .22*swing, (.63-.10*lift)*side, 0.])
                pts = np.array([anchor,knee,foot]) @ rotation * scale + center
                self.legs[3*side_idx+j].set_points_as_corners(pts)
                self.knees[3*side_idx+j].set_width(.046*scale).move_to(pts[1])
        for j, side in enumerate((-1,1)):
            wiggle = .045*math.sin(.63*phase+j)
            pts = np.array([[.59,.09*side,0],[.83,(.25+wiggle)*side,0],[1.0,(.37+wiggle)*side,0]])
            self.antennae[j].set_points_smoothly(pts @ rotation * scale + center)
        return self


def crumbs(at: Point, scale: float = 1.) -> VGroup:
    group = VGroup()
    for x,y,r,a in ((0,0,.14,.1),(.19,.11,.11,.6),(-.14,.14,.08,.8),(.09,-.18,.08,.3),(-.14,-.08,.09,.5)):
        angles = np.linspace(0,2*np.pi,6)[:-1]+a
        poly=Polygon(*(np.array([x+r*np.cos(t),y+r*np.sin(t),0]) for t in angles))
        poly.set_fill(GOLD,.95).set_stroke(WHITE,.5,.5)
        group.add(poly)
    return group.scale(scale).move_to(at)


class Arena(VGroup):
    def __init__(self, motion: Motion, center: Point = ORIGIN, scale: float = 1., ant_scale: float = .60, color: str = BLUE):
        super().__init__()
        self.motion=motion
        self.center=np.asarray(center,dtype=float)
        self.space=scale
        self.color=color
        xs=np.linspace(-6.25,6.25,240)
        points=np.column_stack([xs,trail(xs),np.zeros_like(xs)])*scale+self.center
        self.scent=VGroup(line_curve(points,TEAL,18,.035),line_curve(points,TEAL,8,.055),line_curve(points,TEAL,2,.32))
        self.marks=VGroup(*(
            Dot(p,radius=.017*scale).set_color(TEAL).set_opacity(.48) for p in points[::6]
        ))
        xy=motion.xy
        full=np.column_stack([xy,np.zeros(len(xy))])*scale+self.center
        self.full_track=line_curve(full,color,2.5,.72)
        self.track=self.full_track.copy().pointwise_become_partial(self.full_track,0,.001)
        self.ant=Ant(color,ant_scale)
        self.food=crumbs(self.center+np.array([5.9,float(trail(5.9)),0])*scale,.72*scale)
        self.wind=VGroup(*(Arrow(ORIGIN,UP*.5,buff=0,stroke_width=1.8).set_color(RED) for _ in range(6)))
        self.add(self.scent,self.marks,self.track,self.food,self.ant,self.wind)
        self.set_time(0)

    def point(self, x: float, y: float) -> np.ndarray:
        return self.center+self.space*np.array([x,y,0.])

    def set_time(self, t: float) -> "Arena":
        pose=self.motion.pose(t)
        self.ant.pose_at(self.point(pose.x,pose.y),pose.heading,pose.phase)
        self.track.pointwise_become_partial(self.full_track,0,max(.001,min(1,t/self.motion.seconds)))
        for i,arrow in enumerate(self.wind):
            x=-3.6+i*1.65
            flow=(t*1.15+i*.27)%1.4-.7
            start=self.point(x,flow)
            size=.38+abs(pose.push)*.18
            end=start+np.array([0,math.copysign(size,pose.push or 1),0])
            arrow.put_start_and_end_on(start,end)
            clear_of_ant=np.linalg.norm(start[:2]-self.point(pose.x,pose.y)[:2])>1.05*self.ant.unit
            arrow.set_opacity(.50 if abs(pose.push)>.1 and clear_of_ant else 0.)
        return self

    def follow(self, clock: Callable[[], float]) -> "Arena":
        self.add_updater(lambda mob: mob.set_time(clock()))
        return self


def ruler(at: Point, width: float = 4.0, lo: float = -1.0, hi: float = 1.0, color: str = GRAY) -> VGroup:
    at=np.asarray(at,dtype=float)
    body=Line(at+LEFT*width/2,at+RIGHT*width/2).set_stroke(color,1.5,.8)
    ticks=VGroup()
    for val in np.linspace(lo,hi,5):
        p=at+RIGHT*((val-lo)/(hi-lo)-.5)*width
        ticks.add(Line(p+.07*DOWN,p+.07*UP).set_stroke(color,1.5))
        ticks.add(eq(f"{val:g}",p+.28*DOWN,23,color))
    return VGroup(body,ticks)


class AntFilm(Scene):
    chapter=1

    def setup(self) -> None:
        super().setup()
        self.camera.background_color=BG
        self.started=self.time
        self.duration=DURATIONS[self.chapter-1]
        audio=ROOT/f"communication/video/rendered/v2/audio/s{self.chapter:02}.json"
        self.speech=json.loads(audio.read_text()) if audio.exists() else None
        # One unobtrusive scope label, not a slide title or a narration subtitle.
        self.scope=text("TOY ILLUSTRATION",[-5.65,-3.72,0],12,DIM)
        self.add(self.scope)

    @property
    def elapsed(self) -> float:
        return float(self.time-self.started)

    def at(self,t:float) -> None:
        gap=t-self.elapsed
        if gap>.002:
            self.wait(gap)

    def cue(self,index:int,offset:float=0) -> None:
        if self.speech and index<len(self.speech["boundaries"]):
            data=self.speech
            self.at(data["onset"]+data["boundaries"][index]["start"]/data["tempo"]+offset)

    def finish(self) -> None:
        if self.elapsed>self.duration+.07:
            raise RuntimeError(f"{type(self).__name__} overran: {self.elapsed:.3f} > {self.duration}")
        self.at(self.duration)
