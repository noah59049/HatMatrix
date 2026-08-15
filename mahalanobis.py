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
        scatter_axes = Axes(
            x_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            y_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            x_length=6,
            y_length=6,
            tips=False,
        )
        scatter_dots = VGroup(*[
            Dot(scatter_axes.c2p(x, y), radius=0.05, color=BLUE_C)
            for x, y in samples
        ])
        scatter_group = VGroup(scatter_axes, scatter_dots)

        # Reuse scatter_axes' own x/y mapping for the surface's base plane
        # (rather than a separate ThreeDAxes) so the scatterplot and the
        # density bump agree exactly on where each point sits -- the
        # scatter is literally the surface's z=0 slice.
        HEIGHT_SCALE = 9.0
        density_surface = Surface(
            lambda u, v: scatter_axes.c2p(u, v) + OUT * pdf(u, v) * HEIGHT_SCALE,
            u_range=[-PLOT_RADIUS, PLOT_RADIUS],
            v_range=[-PLOT_RADIUS, PLOT_RADIUS],
            resolution=(32, 32),
            fill_opacity=0.85,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.5,
        )
        
        with self.voiceover("Now let's suppose we have a distribution Q in R^k. This can either be a discrete distribution, which is basically sampling from a set of points,") as tracker:
            quarter = self.get_current_voiceover_duration() / 4
            self.play(FadeIn(scatter_group), run_time=quarter)
        with self.voiceover("or a continuous distribution with a PDF.") as tracker:
            self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=quarter)
            self.play(
                scatter_dots.animate.set_opacity(0.15),
                FadeIn(density_surface),
                run_time=quarter,
            )
            self.play(
                Rotate(density_surface, angle=PI / 2, axis=OUT, about_point=scatter_axes.get_origin()),
                run_time=quarter,
            )
            self.play(
                FadeOut(density_surface),
                scatter_dots.animate.set_opacity(1.0),
            )
            self.move_camera(phi=0, theta=-90 * DEGREES, run_time=0.6)
            self.play(FadeOut(scatter_group), run_time=0.6)

        with self.voiceover("The Mahalanobis distance is a measure of the standardized distance between a point and the mean of the distribution.") as tracker:
            ...
        with self.voiceover("It's given by this formula.") as tracker:
            ...
        with self.voiceover("I think that the best way to explain that formula is to lead with two important properties that Mahalanobis distance should have, and then show how those properties necessitate this formula.") as tracker:
            ...
        with self.voiceover("The first property is that Mahalanobis distance is equal to Euclidean distance from the mean if the covariance matrix is equal to the identity matrix.") as tracker:
            ...
        with self.voiceover("The second property is that rotating or reflecting the entire distribution should not change the Mahalanobis distance of any of the points, and neither should scaling by a nonzero amount along the coordinate axes.") as tracker:
            ...
        with self.voiceover("This property seems intuitive, rotation and reflection should preserve notations of distance. Scaling the axes is also equivalent to changing the units, say measuring in meters instead of centimeters. It's intuitive that that shouldn't change any measures of distance.") as tracker:
            ...
        with self.voiceover("That's it. That's all you need to derive the formula for Mahalanobis distance.") as tracker: # TODO: Change the voiceover?
            ...
        with self.voiceover("So first, any invertible matrix can be written as a product of rotations and scaling the axes. I'm not going to prove this, but you can Google singular value decomposition if you're interested, and here we're just looking at the narrow case of real-valued square matrices.") as tracker:
            ...
        with self.voiceover("So therefore, Mahalanobis distance doesn't change when you multiply the distribution by any matrix.") as tracker:
            ...
