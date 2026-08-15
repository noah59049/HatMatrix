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