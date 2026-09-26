from manimlib import *


FONT = "Noto Serif CJK SC"

MODEL_BLUE = BLUE_C
UPDATE_YELLOW = YELLOW_C
GOOD_GREEN = GREEN_C
HARM_RED = RED_C
MUTED = GREY_B


def make_ant() -> VGroup:
    body = Ellipse(width=0.72, height=0.36)
    head = Circle(radius=0.16).next_to(body, RIGHT, buff=-0.02)
    abdomen = Circle(radius=0.22).next_to(body, LEFT, buff=-0.05)

    antennae = VGroup(
        Line(head.get_right(), head.get_right() + 0.28 * UR),
        Line(head.get_right(), head.get_right() + 0.28 * DR),
    )
    legs = VGroup(
        Line(body.get_center() + 0.10 * LEFT, body.get_center() + 0.34 * UL),
        Line(body.get_center() + 0.10 * LEFT, body.get_center() + 0.34 * DL),
        Line(body.get_center() + 0.10 * RIGHT, body.get_center() + 0.34 * UR),
        Line(body.get_center() + 0.10 * RIGHT, body.get_center() + 0.34 * DR),
    )

    ant = VGroup(abdomen, body, head, antennae, legs)
    ant.set_stroke(MODEL_BLUE, width=3)
    ant.set_fill(MODEL_BLUE, opacity=0.15)
    return ant


class StoryPrototype(Scene):
    def construct(self):
        self.camera.background_color = "#090B10"

        ant = make_ant().scale(0.85).shift(4.7 * LEFT + 0.7 * DOWN)
        food = Dot(4.6 * RIGHT + 0.7 * DOWN, radius=0.16, color=GOOD_GREEN)
        food_ring = Circle(radius=0.38, color=GOOD_GREEN).move_to(food)

        trail = DashedLine(
            ant.get_center() + 0.5 * RIGHT,
            food.get_center() + 0.5 * LEFT,
            dash_length=0.12,
        )
        trail.set_stroke(MUTED, width=2, opacity=0.45)

        opening = Text(
            "一个模型已经会行动。",
            font=FONT,
            font_size=42,
        ).to_edge(UP)

        # checkpoint: familiar-world
        self.play(FadeIn(opening), FadeIn(ant), ShowCreation(trail), FadeIn(food_ring), FadeIn(food))
        self.play(ant.animate.shift(3.2 * RIGHT), run_time=2.0)
        self.wait(0.5)

        hazard = Circle(radius=0.62, color=HARM_RED)
        hazard.set_fill(HARM_RED, opacity=0.12)
        hazard.move_to(0.2 * RIGHT + 0.7 * DOWN)

        changed = Text(
            "环境变了。",
            font=FONT,
            font_size=48,
            color=HARM_RED,
        ).to_edge(UP)

        # checkpoint: environment-change
        self.play(Transform(opening, changed), FadeIn(hazard))
        self.play(ant.animate.shift(0.75 * RIGHT), run_time=0.8)
        self.wait(0.5)

        question = Text(
            "模型应该怎么改变自己？",
            font=FONT,
            font_size=48,
        ).to_edge(UP)

        # checkpoint: central-question
        self.play(Transform(opening, question))
        self.wait(0.8)

        frozen = Text("永远不改", font=FONT, font_size=34, color=MUTED)
        always = Text("每次都改", font=FONT, font_size=34, color=HARM_RED)
        frozen.move_to(3.2 * LEFT + 0.3 * UP)
        always.move_to(3.2 * RIGHT + 0.3 * UP)

        left_arrow = Arrow(0.2 * LEFT, 2.2 * LEFT, buff=0.1, color=MUTED)
        right_arrow = Arrow(0.2 * RIGHT, 2.2 * RIGHT, buff=0.1, color=HARM_RED)

        # checkpoint: two-obvious-answers
        self.play(
            FadeOut(trail),
            FadeOut(food),
            FadeOut(food_ring),
            FadeOut(hazard),
            ant.animate.move_to(ORIGIN + 1.15 * DOWN).scale(0.85),
        )
        self.play(GrowArrow(left_arrow), GrowArrow(right_arrow), FadeIn(frozen), FadeIn(always))
        self.wait(0.8)

        self.play(
            FadeOut(ant),
            FadeOut(left_arrow),
            FadeOut(right_arrow),
            FadeOut(frozen),
            FadeOut(always),
        )

        theta = Dot(1.8 * LEFT + 0.7 * DOWN, radius=0.14, color=MODEL_BLUE)
        theta_label = Text("θ", font=FONT, font_size=38).next_to(theta, DOWN)
        delta = Arrow(
            theta.get_center(),
            theta.get_center() + 3.2 * RIGHT + 1.0 * UP,
            buff=0.16,
            color=UPDATE_YELLOW,
        )
        delta_label = Text("Δθ", font=FONT, font_size=38, color=UPDATE_YELLOW).next_to(delta, UP)

        update_question = Text(
            "这一次更新，值得相信吗？",
            font=FONT,
            font_size=46,
        ).to_edge(UP)

        # checkpoint: update-question
        self.play(Transform(opening, update_question), FadeIn(theta), FadeIn(theta_label))
        self.play(GrowArrow(delta), FadeIn(delta_label))
        self.wait(0.8)

        end_point = Dot(delta.get_end(), radius=0.14, color=UPDATE_YELLOW)
        accept = Text("接受", font=FONT, font_size=32, color=GOOD_GREEN)
        skip = Text("跳过", font=FONT, font_size=32, color=MUTED)
        accept.next_to(end_point, RIGHT, buff=0.35)
        skip.next_to(theta, LEFT, buff=0.35)

        # checkpoint: accept-or-skip
        self.play(FadeIn(end_point), FadeIn(accept), FadeIn(skip))
        self.wait(0.8)

        thesis = Text(
            "不是只学习如何行动，\n而是学习何时允许自己学习。",
            font=FONT,
            font_size=40,
            t2c={"学习何时允许自己学习": UPDATE_YELLOW},
        )
        thesis.move_to(ORIGIN + 0.3 * UP)

        # checkpoint: thesis
        self.play(
            FadeOut(theta),
            FadeOut(theta_label),
            FadeOut(delta),
            FadeOut(delta_label),
            FadeOut(end_point),
            FadeOut(accept),
            FadeOut(skip),
            FadeOut(opening),
        )
        self.play(FadeIn(thesis))
        self.wait(2.0)
