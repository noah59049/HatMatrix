import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class LeveragePhysics(StitcherScene):
    def construct_scene(self):
        leverage_formula = MathTex(
            r"H_{ii}", "=", r"\frac{1}{n}", "+", r"\frac{1}{n} z", "^2"
        ).shift(UP * 2)
        one_over_n, plus, one_over_n_z, squared = leverage_formula[2:]

        torque_formula = Tex(r"Torque = Force $\times$ r").next_to(leverage_formula, DOWN)

        self.add(leverage_formula, torque_formula)

        # Lever: a plank balanced on a fulcrum, with a force pushing down a distance r
        # from the pivot. The plank's tilt is driven by `theta`.
        pivot = DOWN * 1.5
        half_length = 4
        theta = ValueTracker(0)
        r = ValueTracker(0.6)

        def direction():
            return np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])

        fulcrum = Triangle(color=GREY, fill_opacity=1).scale(0.35)
        fulcrum.move_to(pivot, aligned_edge=UP)
        plank = always_redraw(lambda: Line(
            pivot - half_length * direction(),
            pivot + half_length * direction(),
            stroke_width=10,
            color=GREY_B,
        ))
        force_arrow = always_redraw(lambda: Arrow(
            pivot + r.get_value() * direction() + UP * 1.5,
            pivot + r.get_value() * direction(),
            buff=0.1,
            color=RED,
        ))
        r_brace = always_redraw(lambda: BraceBetweenPoints(
            pivot, pivot + r.get_value() * direction(), direction=rotate_vector(direction(), -PI / 2)
        ))
        r_label = always_redraw(lambda: MathTex("r").move_to(
            r_brace.get_tip() + 0.3 * rotate_vector(direction(), -PI / 2)
        ))

        self.play(FadeIn(fulcrum), Create(plank))
        self.play(GrowArrow(force_arrow), FadeIn(r_brace, r_label))

        # Pushing near the center barely tips the lever...
        self.play(theta.animate.set_value(-0.06), rate_func=there_and_back, run_time=2)
        # ...while the same force out at the end swings it hard.
        self.play(r.animate.set_value(3.5), run_time=1.5)
        self.play(theta.animate.set_value(-0.35), rate_func=there_and_back, run_time=2)
        self.wait()

        self.play(Circumscribe(one_over_n_z, Circle, fade_out=True, color=YELLOW))
        self.play(Circumscribe(VGroup(one_over_n, plus), Circle, fade_out=True, color=YELLOW))
        self.play(Circumscribe(squared, Circle, fade_out=True, color=YELLOW))
        self.wait()
