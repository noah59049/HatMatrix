import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, n, rng
from N_Tools import *
from colors import *

# Second predictor, same length as X1. Chosen arbitrarily.
X2 = as_col(np.array([2.1, 1.0, 3.2, 1.8, 2.6, 0.9, 2.4, 3.5, 1.4, 2.9, 2.0, 4.6]))
X12 = np.column_stack([X1, X2])


def get_leverages(X):
    return np.diagonal(X @ np.linalg.inv(X.T @ X) @ X.T)


class Leverages3d(StitcherScene, ThreeDScene):
    def construct_scene(self):
        self.renderer.camera.should_apply_shading = False

        ############################################################
        ### Part 1: The value trackers and always_redraw objects ###
        ############################################################
        # --- ArrayValueTrackers ---
        # X12_tracker holds just the two predictor columns (n x 2), X_tracker
        # prepends the intercept column, same as X1_tracker/X_tracker in
        # leverages_explorer.py.
        X12_tracker = ArrayValueTracker(X12)
        X_tracker = ArrayValueTracker(np.column_stack([as_col(np.ones(n)), X12]))
        X_tracker.add_updater(lambda m: m.set_value(np.column_stack([as_col(np.ones(n)), X12_tracker.get_value()])))
        leverages_tracker = ArrayValueTracker(get_leverages(X_tracker.get_value()))
        leverages_tracker.add_updater(lambda m: m.set_value(get_leverages(X_tracker.get_value())))
        self.add(X12_tracker, X_tracker, leverages_tracker)

        # --- Axes ---
        # One plain Axes for the (x1, x2) base plane; leverage is drawn as
        # height (OUT) above it. Plain Axes rather than ThreeDAxes because
        # ThreeDAxes crashes under FadeIn in this manim version (see
        # X_span.py's strip_sheen and mahalanobis.py).
        X_RANGE = [-0.5, 5.5, 1]
        AXES_LENGTH = 6
        HEIGHT_SCALE = 3.0  # scene units per unit of leverage

        base_axes = Axes(
            x_range = X_RANGE,
            y_range = X_RANGE,
            x_length = AXES_LENGTH,
            y_length = AXES_LENGTH,
            tips = False,
        )
        base_axis_labels = base_axes.get_axis_labels(x_label = "x_1", y_label = "x_2")

        def c2p3(x1, x2, leverage = 0):
            return base_axes.c2p(x1, x2) + OUT * leverage * HEIGHT_SCALE

        x_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot(c2p3(x1, x2), color = X_COLOR) for x1, x2 in X12_tracker.get_value()
                ]
            )
        )
        leverages_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot3D(c2p3(x1, x2, leverage), radius = 0.06, color = H_COLOR)
                    for (x1, x2), leverage in zip(X12_tracker.get_value(), leverages_tracker.get_value().flatten())
                ]
            )
        )
        # Vertical stems from each X dot up to its leverage, so it's clear
        # which height belongs to which point once the camera moves.
        leverage_stems = always_redraw(
            lambda: VGroup(
                *[
                    DashedLine(c2p3(x1, x2), c2p3(x1, x2, leverage), color = GRAY).set_opacity(0.5)
                    for (x1, x2), leverage in zip(X12_tracker.get_value(), leverages_tracker.get_value().flatten())
                ]
            )
        )

        # --- Paraboloid ---
        # The 3D analogue of the parabola in leverages_explorer.py: fit a full
        # quadratic in (x1, x2) to the leverages.
        def quadratic_features(x1, x2):
            return np.column_stack([np.ones_like(x1), x1, x2, x1 * x1, x1 * x2, x2 * x2])

        def get_paraboloid_bhat():
            X12_curr = X12_tracker.get_value()
            paraboloid_X = quadratic_features(X12_curr[:, 0], X12_curr[:, 1])
            paraboloid_Y = as_col(leverages_tracker.get_value())
            paraboloid_bhat = np.linalg.inv(paraboloid_X.T @ paraboloid_X) @ paraboloid_X.T @ paraboloid_Y
            return paraboloid_bhat.flatten()

        def paraboloid_yhat(x1, x2):
            return (quadratic_features(np.atleast_1d(x1), np.atleast_1d(x2)) @ get_paraboloid_bhat())[0]

        def paraboloid_vertex():
            # y = c + b1 x1 + b2 x2 + a11 x1^2 + a12 x1 x2 + a22 x2^2
            #   = c + b.x + x^T A x, with A = [[a11, a12/2], [a12/2, a22]]
            # grad y = b + 2 A x = 0  iff  x = -A^{-1} b / 2
            c, b1, b2, a11, a12, a22 = get_paraboloid_bhat()
            A = np.array([[a11, a12 / 2], [a12 / 2, a22]])
            b = np.array([b1, b2])
            vertex_x = -np.linalg.solve(A, b) / 2
            vertex_y = c + b @ vertex_x + vertex_x @ A @ vertex_x
            return np.array([*vertex_x, vertex_y])

        paraboloid_surface = always_redraw(
            lambda: Surface(
                lambda u, v: c2p3(u, v, paraboloid_yhat(u, v)),
                u_range = [X12_tracker.get_value()[:, 0].min(), X12_tracker.get_value()[:, 0].max()],
                v_range = [X12_tracker.get_value()[:, 1].min(), X12_tracker.get_value()[:, 1].max()],
                resolution = (16, 16),
                fill_opacity = 0.5,
                checkerboard_colors = [BLUE_D, BLUE_E],
                stroke_width = 0.5,
            )
        )
        vertex_dot = always_redraw(lambda: Dot3D(c2p3(*paraboloid_vertex()), radius = 0.06))

        def get_vertex_text():
            v = paraboloid_vertex()
            return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"
        # Labels are fixed in frame (see Part 3) so they stay flat and
        # legible while the camera is tilted.
        vertex_label = always_redraw(lambda: MathTex(get_vertex_text()).to_corner(UR))

        mean_line = always_redraw(
            lambda: DashedLine(
                c2p3(*X12_tracker.get_value().mean(axis = 0), 0),
                c2p3(*X12_tracker.get_value().mean(axis = 0), 1/3),
                color = GRAY
            ).set_opacity(0.5)
        )

        def get_mahalanobis_sq():
            # Squared Mahalanobis distance of each point from the mean, using
            # the population covariance (divide by n) like get_variance in
            # leverages_explorer.py.
            X12_curr = X12_tracker.get_value()
            centered = X12_curr - X12_curr.mean(axis = 0)
            cov = centered.T @ centered / n
            return np.einsum("ij,jk,ik->i", centered, np.linalg.inv(cov), centered)

        #######################################
        ### Part 2: Defining the animations ###
        #######################################
        def dot_dance(number, run_time = 1):
            X12_curr = X12_tracker.get_value()
            mu = X12_curr.mean(axis = 0)
            if   number == 0: # Reset
                new_value = X12.copy()
            elif number == 1: # Pull the outlier in
                new_value = X12_curr.copy()
                new_value[-1] -= np.array([2.5, 1.5])
            elif number == 2: # Shift
                new_value = X12_curr + np.array([0.5, -0.5])
            elif number == 3: # Expand
                new_value = (X12_curr - mu) * 1.5 + mu
            elif number == 4: # Contract
                new_value = (X12_curr - mu) * 0.5 + mu
            elif number == 5: # Rotate about the mean
                angle = PI / 4
                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                new_value = (X12_curr - mu) @ R.T + mu
            elif number == 6: # Stretch along x1 only
                new_value = (X12_curr - mu) * np.array([1.6, 1.0]) + mu
            elif number < 10: # Random correlated values
                cov = np.array([[0.6, 0.3 * (number - 7)], [0.3 * (number - 7), 0.6]])
                new_value = rng.multivariate_normal(mean = [2.5, 2.5], cov = cov, size = n)
            else:
                raise ValueError("dot_dance got an invalid number")

            if run_time == 0:
                X12_tracker.set_value(new_value)
            else:
                self.play(X12_tracker.animate.set_value(new_value), run_time = run_time)

        ##############################
        ### Part 3: Play the scene ###
        ##############################
        # --- The actual scene being played ---
        self.set_camera_orientation(phi = 65 * DEGREES, theta = -60 * DEGREES)

        with self.voiceover("Now let's add a second X variable. Our X values now live in a plane,") as tracker:
            self.add(
                base_axes,
                base_axis_labels,
                x_plot,
            )

        with self.voiceover("and we'll plot each point's leverage as its height above that plane.") as tracker:
            self.add(
                leverage_stems,
                leverages_plot,
            )

        with self.voiceover("In one dimension the leverages fell on a parabola. In two dimensions, let's see if they fall on a paraboloid.") as tracker:
            self.play(IntroduceRedraw(paraboloid_surface, Create))

        with self.voiceover("Let's move the Xs around and see if that holds up.") as tracker:
            self.begin_ambient_camera_rotation(rate = 0.1)

        for i in range(10):
            dot_dance(i, run_time = 0.5)

        with self.voiceover("Where's the vertex?") as tracker:
            self.add_fixed_in_frame_mobjects(vertex_label)
            self.add(vertex_dot, mean_line)

        for i in range(10):
            dot_dance(i, run_time = 0.5)

        with self.voiceover("TODO") as tracker:
            self.stop_ambient_camera_rotation()

# TODO: Connect the leverages to the Mahalanobis distance:
# H_ii = 1/n + (squared Mahalanobis distance) / n, via get_mahalanobis_sq().
