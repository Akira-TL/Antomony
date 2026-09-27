"""Question-led stage cut. New scenes only; approved v2 core clips are reused.

Use scripts/video/stage.sh interactive SCENE draft LINE for checkpoint editing.
No caption rendering, copied third-party footage, or main-project model calls.
"""
from __future__ import annotations

import math
import numpy as np
from manimlib import *

from communication.video.stage.plan import CHAPTERS
from communication.video.stage.network import LivingNetwork
from communication.video.scenes.ant_drawing import (
    Ant, Arena, BG, WHITE, BLUE, GOLD, TEAL, RED, GRAY, DIM,
    text, eq, ruler, line_curve,
)
from communication.video.scenes.ants_learning import steering_walk, numeric
from communication.video.production.toy_motion import rollout, trail


class StageScene(Scene):
    chapter = 0

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BG
        self.started = self.time
        self.duration = CHAPTERS[self.chapter].seconds
        self.scope = text("TOY ILLUSTRATION", [-5.65, -3.72, 0], 12, DIM)
        self.add(self.scope)

    @property
    def elapsed(self) -> float:
        return float(self.time - self.started)

    def at(self, t: float) -> None:
        if t - self.elapsed > .002:
            self.wait(t - self.elapsed)

    def finish(self) -> None:
        if self.elapsed > self.duration + .06:
            raise RuntimeError(f"{type(self).__name__}: {self.elapsed} exceeds {self.duration}")
        self.at(self.duration)


def opening_world() -> Arena:
    """Shared exact geometry and simulation clock at the 30-second splice."""
    return Arena(rollout(18., "fixed", "shift"), ant_scale=.86)


class B00TheQuestion(StageScene):
    chapter = 0

    def construct(self) -> None:
        self.remove(self.scope)
        self.scope = text("TRAIN-THEN-DEPLOY EXAMPLE", [-5.20, -3.72, 0], 12, DIM)
        self.add(self.scope)
        net = LivingNetwork(np.array([-2.30, .15, 0]))
        net.add_updater(lambda n: n.update_at(self.elapsed))
        self.add(net)

        # checkpoint: activity-continues-after-the-weights-stop
        dial_centers = [np.array([x, 2.25, 0]) for x in (-4.3, -2.6, -.9)]
        identities = ((0, 1, 0), (1, 2, 1), (2, 0, 2))
        dials = VGroup()
        needles = VGroup()
        for i, (p, identity) in enumerate(zip(dial_centers, identities), 1):
            circle = Circle(radius=.25).move_to(p).set_stroke(GRAY, 1.2, .8)
            ticks = VGroup(*(Line(p + .21 * np.array([np.cos(a), np.sin(a), 0]),
                                  p + .29 * np.array([np.cos(a), np.sin(a), 0])).set_stroke(GRAY, 1)
                             for a in np.linspace(-PI / 4, 5 * PI / 4, 5)))
            needle = Line(p, p + RIGHT * .20).set_stroke(GOLD, 2.5)
            def turn(mob: Line, dt: float, point=p, index=identity) -> None:
                value = float(net.parameters(self.elapsed)[index[0]][index[1], index[2]])
                angle = PI * .75 - value * .95
                mob.put_start_and_end_on(point, point + .21 * np.array([np.cos(angle), np.sin(angle), 0]))
            needle.add_updater(turn)
            dials.add(circle, ticks, eq(rf"w_{i}", p + .49 * DOWN, 27, GRAY))
            needles.add(needle)
        self.add(dials, needles)
        done = text("Training finished.", [-3.3, 3.15, 0], 34)
        question = text("Learning finished?", [3.1, 3.15, 0], 36, GOLD)
        self.play(FadeIn(done), run_time=.65)
        self.at(1.95)
        self.play(FadeIn(question, .10 * UP), run_time=.70)

        # Predicted and observed curves use a fixed network after training.
        axes = VGroup(Line([2., -1.05, 0], [6.15, -1.05, 0]),
                      Line([2., -1.05, 0], [2., 1.65, 0])).set_stroke(DIM, 1.3)
        legends = VGroup(text("prediction", [2.95, 2.1, 0], 23, BLUE),
                         text("observation", [5.10, 2.1, 0], 23, WHITE))
        predicted = line_curve(np.array([[2., 0, 0], [2.01, 0, 0]]), BLUE, 2.8)
        observed = predicted.copy().set_stroke(WHITE, 2.4)
        p_dot = Dot(radius=.065).set_color(BLUE)
        c_dot = Dot(radius=.065).set_color(WHITE)
        residual = Line(ORIGIN, UP * .03).set_stroke(RED, 3)
        graph = VGroup(predicted, observed, residual, p_dot, c_dot)
        def update_graph(group: VGroup) -> None:
            end = max(.10, self.elapsed - .32)  # observations arrive after predictions
            begin = max(0., end - 6.0)
            ts = np.linspace(begin, end, 75)
            px = 2.1 + 3.90 * (ts - begin) / 6.0
            py = np.array([net.prediction(float(s)) for s in ts]) * 1.15 + .15
            cy = py + .65 * (ts >= 8.15)
            predicted.set_points_as_corners(np.column_stack([px, py, np.zeros_like(px)]))
            observed.set_points_as_corners(np.column_stack([px, cy, np.zeros_like(px)]))
            p_dot.move_to([px[-1], py[-1], 0])
            c_dot.move_to([px[-1], cy[-1], 0])
            residual.put_start_and_end_on(p_dot.get_center(), c_dot.get_center() + [.0, .002, 0])
            residual.set_opacity(1. if self.elapsed > 8.47 else 0.)
        graph.add_updater(update_graph)
        formula = eq(r"\hat y_t=f_\theta(x_t)", [4.05, -1.80, 0], 49)
        formula[r"\theta"].set_color(GOLD)
        self.at(4.45)
        self.play(FadeIn(axes), FadeIn(legends), Write(formula), run_time=.95)
        self.add(graph)
        self.at(8.50)
        self.play(FlashAround(c_dot, color=RED), run_time=.70)
        self.at(10.7)
        self.play(FadeOut(done), FadeOut(question), FadeOut(dials), FadeOut(needles),
                  run_time=.75)

        # checkpoint: global-dependency-versus-one-local-candidate
        back_paths = []
        for layer in reversed(net.edge_layers):
            bright = layer.copy().set_stroke(RED, 2.2, .58)
            for edge in bright:
                edge.reverse_points()
            back_paths.append(ShowPassingFlash(bright, time_width=.45))
        self.play(LaggedStart(*back_paths, lag_ratio=.30), run_time=1.35)
        edge = net.selected_edge()
        local = Ellipse(width=2.65, height=2.55).move_to(edge.get_center()).set_stroke(GOLD, 1.6, .85)
        candidate = DashedLine(edge.get_start() + .15 * UP, edge.get_end() + .15 * UP,
                               dash_length=.085).set_stroke(GOLD, 3.2)
        delta = eq(r"\Delta\theta_{\mathrm{local}}", [-.62, -2.0, 0], 49, GOLD)
        self.play(ShowCreation(local), ShowCreation(candidate), Write(delta), run_time=1.15)
        self.at(17.5)
        graph.clear_updaters()
        self.play(FadeOut(graph), FadeOut(axes), FadeOut(legends), FadeOut(formula), run_time=.65)
        choices = text("Which changes?", [3.30, 1.65, 0], 35, GOLD)
        self.play(FadeIn(choices), run_time=.60)
        baseline = Line([-5.45, -2.95, 0], [5.50, -2.95, 0]).set_stroke(DIM, 1.2)
        beats = VGroup(*(Dot([x, -2.95, 0], radius=.045).set_color(BLUE)
                         for x in np.linspace(-5.3, 5.3, 15)))
        pointer = Dot(radius=.085).set_color(TEAL)
        pointer.add_updater(lambda d: d.move_to([-5.3 + (self.elapsed * 1.4) % 10.6, -2.95, 0]))
        action_label = text("actions keep coming", [-3.9, -2.58, 0], 22, BLUE)
        self.play(ShowCreation(baseline), FadeIn(beats), FadeIn(action_label), run_time=.60)
        self.add(pointer)
        self.at(21.4)
        deadline = text("Before the next action?", [3.12, 1.65, 0], 31, WHITE)
        self.play(Transform(choices, deadline), run_time=.65)

        # checkpoint: the-question-enters-the-ant-world
        self.at(25.)
        net.clear_updaters()
        pointer.clear_updaters()
        world = opening_world()
        pose = world.motion.pose(0.)
        little_center = np.array([pose.x, pose.y, 0])
        self.play(
            net.animate.scale(.11).move_to(little_center),
            FadeOut(local), FadeOut(candidate), FadeOut(delta), FadeOut(choices),
            FadeOut(baseline), FadeOut(beats), FadeOut(pointer), FadeOut(action_label),
            FadeOut(self.scope), run_time=1.65,
        )
        self.play(FadeIn(world.scent), FadeIn(world.marks), FadeIn(world.food),
                  FadeIn(world.ant), FadeOut(net), run_time=.75)
        self.remove(world.scent, world.marks, world.food, world.ant)
        self.scope = text("TOY ILLUSTRATION", [-5.65, -3.72, 0], 12, DIM)
        self.add(self.scope, world)
        world.follow(lambda: max(0., self.elapsed - 28.))
        self.finish()


class B01FollowTheTrail(StageScene):
    chapter = 1

    def construct(self) -> None:
        world = opening_world().follow(lambda: self.elapsed + 2.)
        world.set_time(2.)
        self.add(world)
        # checkpoint: seamless-world-continuation-after-the-introduction
        note = text("a sideways push", [2.6, 2.6, 0], 31, RED)
        self.at(3.05)
        self.play(FadeIn(note, .1 * DOWN), run_time=.55)
        predicted = Ant(GRAY, .86).set_opacity(.22)
        gap = Line(ORIGIN, UP * .1).set_stroke(RED, 2)
        def compare(mob: Ant) -> None:
            p = world.motion.pose(self.elapsed + 2.)
            a = np.array([p.x, float(trail(p.x)), 0])
            b = np.array([p.x, p.y, 0])
            mob.pose_at(a, 0., p.phase)
            gap.put_start_and_end_on(a, b + .001 * UP)
        predicted.add_updater(compare)
        self.at(5.5)
        self.play(FadeIn(predicted), ShowCreation(gap), run_time=.65)
        q = text("Same action. Different outcome.", [0, -2.45, 0], 36)
        self.at(7.2)
        self.play(FadeIn(q), run_time=.65)
        self.at(12.8)
        self.play(FadeOut(note), FadeOut(predicted), FadeOut(gap), run_time=.6)
        self.finish()


class B02TheTwin(StageScene):
    chapter = 2

    def construct(self) -> None:
        # Keep the original physical speed; omit the already-explained late wait.
        top = Arena(rollout(20., "fixed", "shift"), [0, 1.30, 0], .93, .59, GRAY).follow(lambda: self.elapsed)
        bottom = Arena(rollout(20., "always", "shift"), [0, -1.6, 0], .93, .59, TEAL).follow(lambda: self.elapsed)
        names = VGroup(text("fixed steering", [-4.5, 3.02, 0], 29, GRAY),
                       text("local correction", [-4.5, -.38, 0], 29, TEAL))
        self.add(top, bottom, names)
        same = text("same starting controller", [1.5, 3.02, 0], 26)
        self.play(FadeIn(same), run_time=.6)
        self.at(4.5)
        self.play(FadeOut(same), run_time=.45)
        f0 = eq("f=0", [4.1, 3., 0], 42, GRAY)
        f1 = eq("f=", [3.55, -.38, 0], 42, TEAL)
        value = numeric(0, [4.42, -.38, 0], TEAL, 37)
        value.add_updater(lambda m: m.set_value(bottom.motion.pose(self.elapsed).weight))
        self.at(5.9)
        self.play(FadeIn(f0), FadeIn(f1), run_time=.65)
        self.add(value)
        question = text("What changed inside?", [0, -3.25, 0], 32)
        self.at(9.4)
        self.play(FadeIn(question), run_time=.65)
        self.finish()


class B03OneCorrection(StageScene):
    chapter = 3

    def construct(self) -> None:
        ant = Ant(TEAL, 1.26)
        axis = ruler([.4, 1.4, 0], 8., -.5, 1.5)
        sname = eq("s", [-1.6, 2.1, 0], 47, BLUE)
        fname = eq("f", [1.1, 2.1, 0], 47, GOLD)
        fixed = Dot([-1.6, 1.4, 0], radius=.09).set_color(BLUE)
        tip = Triangle().scale(.10).rotate(PI).set_color(GOLD)
        value = numeric(0, [4.5, 2.05, 0], GOLD, 41)
        formula = eq("w=s+f", [3.25, -1.25, 0], 66)
        formula["s"].set_color(BLUE)
        formula["f"].set_color(GOLD)
        force = Arrow(ORIGIN, RIGHT, buff=0, stroke_width=5).set_color(WHITE)
        wind = Arrow(ORIGIN, UP, buff=0, stroke_width=3).set_color(RED)
        correction = Arrow(ORIGIN, DOWN, buff=0, stroke_width=4).set_color(GOLD)
        def drive(mob: Ant) -> None:
            t = self.elapsed
            f = .85 * smooth(float(np.clip((t - 2.9) / 4., 0, 1)))
            p = np.array([-4.65 + .13 * t, -1.25 + .1 * math.sin(t * .6), 0])
            mob.pose_at(p, math.atan2(-f, .85), t * 9)
            start = p + [.85, .1, 0]
            force.put_start_and_end_on(start, start + [1.45, 1.6 * (.85 - f), 0])
            wind.put_start_and_end_on(start + RIGHT * 1.45, start + RIGHT * 1.45 + UP * 1.36)
            correction.put_start_and_end_on(start, start + DOWN * max(.025, f * 1.6))
            correction.set_opacity(1. if f > .01 else 0)
            tip.move_to([4 * f - 1.6, 1.67, 0])
            value.set_value(f)
        ant.add_updater(drive)
        ground = line_curve(np.array([[x, -2.35 + .045 * np.sin(x * 3), 0]
                                      for x in np.linspace(-6.4, .25, 90)]), DIM, 1.4, .7)
        keep = text("basic movement stays", [-3.55, -2.8, 0], 25, BLUE)
        self.add(ground, ant, force, wind, keep)
        self.at(1.4)
        self.play(ShowCreation(axis), FadeIn(fixed), FadeIn(tip), FadeIn(sname), FadeIn(fname), run_time=.95)
        self.add(value, correction)
        # checkpoint: a-single-complete-sweep-from-weight-to-motion
        self.at(7.7)
        self.play(Write(formula), run_time=1.25)
        answer = text("one small correction", [3.2, -2.23, 0], 30, GOLD)
        self.at(11.7)
        self.play(FadeIn(answer), run_time=.7)
        self.finish()


class B07Withdraw(StageScene):
    chapter = 7

    def construct(self) -> None:
        ant, path = steering_walk(self, 3.1, -.74375, -.38125, .60, -1.70, .68, .80)
        self.add(path, ant)
        axis = ruler([0, 1.0, 0], 8., 0, 1)
        left = np.array([-4, 1., 0])
        first = Arrow(left, left + RIGHT * 2.9, buff=0, stroke_width=8).set_color(GOLD)
        second = Arrow(left + RIGHT * 2.9, left + RIGHT * 5.95, buff=0, stroke_width=8).set_color(TEAL)
        d1 = eq(r"\delta_j", [-2.55, 1.8, 0], 52, GOLD)
        d2 = eq(r"\delta_k", [.48, 1.8, 0], 52, TEAL)
        total = Dot(left + RIGHT * 5.95, radius=.09).set_color(WHITE)
        self.add(axis, first, second, d1, d2, total)
        formula = eq(r"F\leftarrow F-r\delta_j", [2.65, 2.7, 0], 51)
        formula[r"\delta_j"].set_color(GOLD)
        self.play(Write(formula), run_time=.9)
        # checkpoint: remove-one-contribution-while-the-ant-keeps-walking
        self.at(1.3)
        self.play(FadeOut(first, UP * .6), FadeOut(d1, UP * .6),
                  second.animate.shift(LEFT * 2.9), d2.animate.shift(LEFT * 2.9),
                  total.animate.shift(LEFT * 2.9), run_time=1.8)
        footprints = VGroup(*(Dot([-5.2 + i * .38, -2.4, 0], radius=.022).set_color(GRAY) for i in range(10)))
        self.play(FadeIn(footprints), run_time=.45)
        past = text("the path already walked", [-2.5, -2.97, 0], 28, GRAY)
        self.at(4.8)
        self.play(FadeIn(past), run_time=.55)
        self.finish()


class B09TheArchitecture(StageScene):
    chapter = 9

    def construct(self) -> None:
        for y, policy, color, name in ((1.7, "fixed", GRAY, "fixed rule"),
                                       (-.2, "always", GOLD, "always update"),
                                       (-2.1, "selective", TEAL, "selective updates")):
            world = Arena(rollout(24., policy, "mixed"), [0, y, 0], .86, .44, color).follow(lambda: self.elapsed + 10.)
            self.add(world, text(name, [-4.8, y + .9, 0], 25, color))
        self.scope.fix_in_frame()
        # Keep the ants walking while the camera pulls back to the larger question.
        net = LivingNetwork(np.array([0, 8.0, 0]), 1.2)
        net.add_updater(lambda n: n.update_at(self.elapsed + 1.3))
        self.add(net)
        self.at(2.35)
        self.play(self.frame.animate.set_height(16.8).move_to([0, 4.4, 0]),
                  run_time=2.45, rate_func=smooth)
        selected = net.selected_edge().copy().set_stroke(GOLD, 4)
        self.play(ShowPassingFlash(selected, time_width=.6), run_time=.65)
        title = text("Learning While Acting", [0, 3.35, 0], 45).fix_in_frame()
        subtitle = text("An architecture proposal · MathHackson", [0, -3.67, 0], 20, GRAY).fix_in_frame()
        self.play(FadeIn(title), FadeOut(self.scope), FadeIn(subtitle), run_time=.65)
        self.finish()
