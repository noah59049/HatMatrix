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

        # Wait actually
        """
        Now would be an excellent time to bring in the physical interpretation of linear regression.
        What I'm doing here is heavily inspired by Joshua Loftus's 2020 blog post "Least squares as springs". You should check it out.
        The points are fixed, say they're nails in a board.
        Each point has a spring attached to it. And this spring has to be vertical, but it can stretch or shrink as much as it wants. If I were to build this in real life, I would probably put a fixed vertical rod in the center of the spring.
        The regression line can be represented as a straight rod that's attached to each of the springs. The springs are attached to the rod, but not glued or bolted. Let's say that there are loops around the springs for the rod.

        The force exerted by each of the springs is equal to k times the distance the spring is stretched. This is Hooke's Law, and k is a constant for the spring, measuring the stiffness.
        We assume that each spring is the same, and, slightly unrealistically, that each spring has resting length zero.
        We're also assuming things like gravity, friction, mass of the springs themselves, etc are negligible.

        So, to be clear, the rod represents the regression line, the nails are the datapoints, and each spring goes from a datapoint to its fitted value, from y to y hat, and each length of the spring is a residual.

        Now, if the system is at equilibrium, three things are true:
        (1) The total potential energy is minimized.
        This is a standard principle from physics; systems, from chemical reactions to balls rolling down a hill, tend to settle in a state that minimizes their potential energy.
        The potential energy of a spring is equal to the integral over distance of the force exerted. Here, the distance is the residual, ei, so the potential energy is the integral from 0 to ei of k times ei.
        So that means the total potential energy is the sum of the potential energies of all the springs, or 1/2 k times the sum of squared lengths of the springs.
        This is, up to multiplying by a constant, exactly equal to the sum of squared errors that least squares regression minimizes! Therefore this setup is the least-squares regression line.

        (2) The net force on the rod is zero. If it weren't, the rod would not stay at rest.
        This means that the sum of the displacement of the springs has to be zero, or in other words, that the sum of the resdiuals has to be zero.
        As a consequence, the rod/regression line has to pass through (x bar, y bar).

        (3) The net torque on the rod is zero.
        Something I learned while animating this video is that if the net force is zero, the net torque is equal around every point. 
        So engineers or people solving these kinds of problems just pick whichever point is most convenient to calculate net torque around.
        For us, that point is the center of the datapoints, (x bar, y bar), which we know the rod passes through.
        Anyway, the total torque is the sum, for each point, of the distance from the center in the x direction, times the residual, times k.
        The k is a constant, so that leaves us with that the sum of the residuals times the distance from the mean in the x-direction must be zero.

        I and the original blog post said that the points were fixed, I even suggested they could be nails. I'm going to break that rule now though because I want to use this physical model to talk about leverages.
        """