import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class Mahalanobis(StitcherScene, ThreeDScene):
    def construct_scene(self):
        self.renderer.camera.should_apply_shading = False

        cov = np.array([[1.4, 0.8], [0.8, 1.0]])
        mean = np.array([0.0, 0.0])
        inv_cov = np.linalg.inv(cov)
        norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

        def pdf(x, y):
            v = np.array([x, y]) - mean
            return norm_const * np.exp(-0.5 * v @ inv_cov @ v)

        rng = np.random.default_rng(3)
        L = np.linalg.cholesky(cov)
        samples = (L @ rng.standard_normal((2, 80))).T + mean

        PLOT_RADIUS = 4
        AXES_LENGTH = 5
        scatter_axes = Axes(
            x_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            y_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            x_length=AXES_LENGTH,
            y_length=AXES_LENGTH,
            tips=False,
        ).to_edge(LEFT, buff=0.75)
        scatter_dots = VGroup(*[
            Dot(scatter_axes.c2p(x, y), radius=0.05, color=BLUE_C)
            for x, y in samples
        ])
        scatter_group = VGroup(scatter_axes, scatter_dots)

        # A separate plain Axes (not ThreeDAxes) just to get an x/y base
        # mapping for the surface, positioned to the right of the
        # scatterplot -- ThreeDAxes crashes under FadeIn in this manim
        # version (see X_span.py's strip_sheen), so we sidestep it entirely
        # rather than work around it here.
        density_axes = Axes(
            x_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            y_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            x_length=AXES_LENGTH,
            y_length=AXES_LENGTH,
            tips=False,
        ).to_edge(RIGHT, buff=0.75)
        HEIGHT_SCALE = 9.0
        density_surface = Surface(
            lambda u, v: density_axes.c2p(u, v) + OUT * pdf(u, v) * HEIGHT_SCALE,
            u_range=[-PLOT_RADIUS, PLOT_RADIUS],
            v_range=[-PLOT_RADIUS, PLOT_RADIUS],
            resolution=(32, 32),
            fill_opacity=0.85,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.5,
        )
        density_group = VGroup(density_axes, density_surface)

        with self.voiceover("Now let's suppose we have a distribution Q in R^k. This can either be a discrete distribution, which is basically sampling from a set of points,") as tracker:
            # Locked to screen space so it stays flat and legible once the
            # camera tilts to show the 3D graph next to it below.
            self.add_fixed_in_frame_mobjects(scatter_group)
            self.play(FadeIn(scatter_group), run_time=self.get_current_voiceover_duration())
        with self.voiceover("or a continuous distribution with a PDF.") as tracker:
            half = self.get_current_voiceover_duration() / 2
            self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=0.6)
            self.play(FadeIn(density_group), run_time=half)
            self.play(FadeOut(density_group), run_time=half)
            self.move_camera(phi=0, theta=-90 * DEGREES)

        formula_tex = MathTex(
            r"D(\vec{x}, Q) = ", 
            r"\sqrt{(\vec{x} - \vec{\mu})^T cov(W)^{-1} (\vec{x} - \vec{\mu})}"
        ).to_corner(UL)

        with self.voiceover("The Mahalanobis distance is a measure of the standardized distance between a point and the mean of the distribution.") as tracker:
            ... # TODO: Draw dotted lines from each point to the mean of the 2D graph, then have a small label with the mahalanobis distance of each point
            # The lines should be always redraw, but the distance numbers should not because we will never change them.
            self.play(Write(formula_tex[0]))
        with self.voiceover("It's given by this formula.") as tracker:
            self.play(Write(formula_tex[1]))

        properties = VGroup(
            MathTex(r"cov(Q) = I \implies D(\vec{x}, Q) = ||\vec{x} - \vec{\mu}||"),
            VGroup(
                MathTex(r"D(\vec{x}, Q) = D(A \vec{x}, A Q)"),
                Tex("if A is a rotation, reflection, or scales along the axes"),
            ).arrange(DOWN, aligned_edge = LEFT)
        ).arrange(DOWN, aligned_edge = LEFT)
        property_numbers = VGroup(*[
            Tex(f"{i}.").next_to(property, LEFT, aligned_edge=UP)
            for i, property in enumerate(properties, start=1)
        ])
        property_numbers.next_to(scatter_axes, RIGHT)
        properties.next_to(property_numbers, RIGHT, aligned_edge=UP)
        with self.voiceover("I think that the best way to explain that formula is to lead with two important properties that Mahalanobis distance should have, and then show how those properties necessitate this formula.") as tracker:
            self.play(FadeIn(property_numbers))
        with self.voiceover("The first property is that Mahalanobis distance is equal to Euclidean distance from the mean if the covariance matrix is equal to the identity matrix.") as tracker:
            self.play(Write(properties[0]))
        with self.voiceover("The second property is that rotating or reflecting the entire distribution should not change the Mahalanobis distance of any of the points, and neither should scaling by a nonzero amount along the coordinate axes.") as tracker:
            self.play(Write(properties[1]))
        with self.voiceover("This property seems intuitive; rotation and reflection should preserve notations of distance.") as tracker:
            ... # TODO: Rotate and reflect the points
        with self.voiceover("Scaling the distribution along the axes is also equivalent to changing the units, say measuring in meters instead of centimeters. It's intuitive that that shouldn't change any measures of distance.") as tracker:
            ... # TODO: Scale the points on the axes
        with self.voiceover("That's it. That's all you need to derive the formula for Mahalanobis distance.") as tracker: # TODO: Change the voiceover?
            ...
        with self.voiceover("So first, any invertible matrix can be written as a product of rotations and scaling the axes. I'm not going to prove this, but you can Google singular value decomposition if you're interested, and here we're just looking at the narrow case of real-valued square matrices.") as tracker:
            ...
        with self.voiceover("So therefore, Mahalanobis distance doesn't change when you multiply the distribution by any matrix.") as tracker:
            ...
