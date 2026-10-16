import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from N_Tools import *
from colors import *

# 8x8 grid of (x1, x2) points, centered at (2.5, 2.5).
GRID_SIDE = 8
grid_1d = np.linspace(1, 4, GRID_SIDE)
X12 = np.array([[x1, x2] for x2 in grid_1d for x1 in grid_1d])
n = len(X12)
SKYSCRAPER_COLOR = GREEN


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
        # Scene units per unit of leverage. Leverages are small with n = 64
        # (around 1/n to 6/n on the grid), so this has to be large.
        HEIGHT_SCALE = 20.0
        SKYSCRAPER_WIDTH = 0.12

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
        # A green "skyscraper" under each leverage dot: a box rising from
        # the X dot on the base plane up to that point's leverage.
        def skyscraper(x1, x2, leverage):
            return Prism(
                dimensions = [SKYSCRAPER_WIDTH, SKYSCRAPER_WIDTH, leverage * HEIGHT_SCALE],
                fill_color = SKYSCRAPER_COLOR,
                fill_opacity = 1,
                stroke_color = SKYSCRAPER_COLOR,
                stroke_width = 0.5,
            ).move_to(c2p3(x1, x2, leverage / 2))

        skyscrapers = always_redraw(
            lambda: VGroup(
                *[
                    skyscraper(x1, x2, leverage)
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
                c2p3(*X12_tracker.get_value().mean(axis = 0), 3/n),
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
        # Every move is a linear map about the mean of the Xs, applied
        # continuously (M(alpha) at each frame) rather than by interpolating
        # start and end positions, so a rotation actually rotates instead of
        # cutting straight through the middle.
        def transform_about_mean(matrix_at, run_time = 1):
            start = X12_tracker.get_value().copy()
            mu = start.mean(axis = 0)
            self.play(
                UpdateFromAlphaFunc(
                    X12_tracker,
                    lambda m, alpha: m.set_value((start - mu) @ matrix_at(alpha).T + mu),
                ),
                run_time = run_time,
            )

        def scale_along(direction, from_factor, to_factor, run_time = 1):
            """Scales along direction from from_factor to to_factor, both
            measured relative to the original grid, so scale_along(d, 2, 1)
            exactly undoes scale_along(d, 1, 2)."""
            d = np.array(direction, dtype = float)
            d = as_col(d / np.linalg.norm(d))
            def matrix_at(alpha):
                factor = from_factor + (to_factor - from_factor) * alpha
                return np.eye(2) + (factor / from_factor - 1) * (d @ d.T)
            transform_about_mean(matrix_at, run_time)

        def rotate_by(angle, run_time = 1):
            def matrix_at(alpha):
                a = angle * alpha
                return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            transform_about_mean(matrix_at, run_time)

        def dot_dance(i, run_time = 1):
            if i == 0:
                scale_along([1, 0], 1, 2, run_time)       # Stretch the x1 axis
            elif i == 1:
                scale_along([1, 0], 2, 1, run_time)       # Shrink it back
            elif i == 2:
                rotate_by(PI / 2, run_time)               # Rotate 90 degrees
            elif i == 3:
                scale_along([1, -1], 1, 0.3, run_time)    # Squish along y = -x
            elif i == 4:
                scale_along([1, -1], 0.3, 1, run_time)    # Move it back
            else:
                raise ValueError(f"dot_dance got incorrect value of i: {i}")

        ##############################
        ### Part 3: Play the scene ###
        ##############################
        # --- The actual scene being played ---
        self.set_camera_orientation(phi = 55 * DEGREES, theta = -60 * DEGREES)

        self.add(
            base_axes,
            base_axis_labels,
            # x_plot,
        )

        self.add(
            skyscrapers,
            leverages_plot,
        )

        return # TODO: Remove

        self.play(IntroduceRedraw(paraboloid_surface, Create))

        self.begin_ambient_camera_rotation(rate = 0.1)

        dot_dance() # TODO: change

        self.add_fixed_in_frame_mobjects(vertex_label)
        self.add(vertex_dot, mean_line)

        dot_dance() # TODO: change

        self.stop_ambient_camera_rotation()

# TODO: Connect the leverages to the Mahalanobis distance:
# H_ii = 1/n + (squared Mahalanobis distance) / n, via get_mahalanobis_sq().
