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

        self.play(Circumscribe(one_over_n_z, Rectangle, fade_out=True, color=YELLOW))
        self.play(Circumscribe(squared, Rectangle, fade_out=True, color=YELLOW))
        self.play(Circumscribe(one_over_n), Rectangle, fade_out=True, color=YELLOW)
        self.wait()


        """
        The term 'leverage' has some relation to leverage in physics.
        Now, a disclaimer before we get started, my knowledge of physics is one class on it in high school, so take my physics stuff with a grain of salt.
        Anyway, the idea of leverage is that if you push on a lever near the fulcrum, it doesn't move that much. But if you push far away from the fulcrum, it moves a lot.
        There's a formula, torque equals force times distance from the fulcrum.
        If you look at our formula for leverages though, it doesn't quite look like that.

        Like, this 1/n I guess can be a proportionality constant, but what about the squared? Or what about the 1/n here?

        The change in slope does basically look like the law of the lever though.

        Here's the formula for slope. If we change one of the yi's, the slope changes by delta yi times the xi's distance from the mean, and divided by a constant, the sum of squared deviations of x.
        If we change one of the yi's that's twice as far from the mean in the x-direction, the slope will change twice as much.
        """