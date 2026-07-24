import numpy as np
from manim import *
from N_Tools import *
from stitcher_scene import StitcherScene


class XSpan(StitcherScene, ThreeDScene):
    def construct_scene(self):
        self.silent = True
        # n = 3, k = 2. Chosen arbitrarily: first column of X is all ones (intercept).
        n, k = 3, 2
        X = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0],
        ])
        # Choose Y so that its OLS fit is exactly bhat_ols: any component of Y
        # orthogonal to X's column span changes the residual but not the
        # projection, so adding get_unit_normal(X0, X1) (perpendicular to both
        # columns, since n - k = 1 here) to X @ bhat_ols leaves the fit alone.
        bhat_ols = np.array([0.3, 0.7])
        Y_residual_scale = 3  # how far off the X0/X1 plane Y sits
        Y = X @ bhat_ols + Y_residual_scale * get_unit_normal(X[:, 0], X[:, 1])

        # Setup the 3D axes with the span of X
        bhat = ArrayValueTracker([0.0, 0.0])

        yhat = ArrayValueTracker(X @ bhat.get_value())
        yhat.add_updater(lambda m: m.set_value(X @ bhat.get_value()))

        self.add(bhat, yhat)

        axes = ThreeDAxes()
        axes.scale(0.7)
        # A dict (not a plain bool) so the lambda below closes over a
        # mutable cell we can flip later, instead of a snapshotted value.
        # We hide/show yhat_point via this opacity flag rather than
        # self.remove()/self.add() (or FadeOut, which removes by default):
        # yhat_point is a submobject of graph_group, and Scene.remove()
        # explicitly *dissolves* the parent group in self.mobjects when you
        # remove one of its children (see restructure_mobjects's docstring).
        # Once dissolved, graph_group stops being a tracked scene entity, so
        # anything added to it later (rotate updater included) never
        # actually gets touched by the scene's update loop again.
        yhat_point_visible = {"value": True}
        yhat_point = always_redraw(lambda: Dot3D(
            axes.c2p(*yhat.get_value()), color=YELLOW, radius=0.1
        ).set_opacity(1.0 if yhat_point_visible["value"] else 0.0))
        y_point = Dot3D(axes.c2p(*Y), color=ORANGE, radius=0.1)
        graph_group = VGroup(axes, yhat_point, y_point)
        
        # Rotate the objects instead of orbiting the camera (camera stays at
        # its default, identity-like pose). ThreeDCamera projects world points
        # via rot_matrix @ (point - frame_center), where
        # rot_matrix = Rz(gamma) @ Rx(-phi) @ Rz(-theta - 90°)
        # (see ThreeDCamera.generate_rotation_matrix). With gamma=0 and the
        # default frame_center=ORIGIN, pre-applying that same matrix to our
        # objects and leaving the camera untouched renders identically to
        # calling self.set_camera_orientation(phi=camera_phi, theta=camera_theta).
        #
        # Note this also keeps axes.c2p() correct for everything built from it
        # afterwards (the always-redrawn point, the X0/X1 arrows, the span
        # plane): apply_matrix rotates the axes' own NumberLines in place, and
        # coords_to_point() reads their current (rotated) geometry, so c2p just
        # keeps working in the rotated frame with no extra bookkeeping.
        camera_phi = 70 * DEGREES
        origin = axes.c2p(0, 0, 0)
        X0_dir = axes.c2p(*X[:, 0]) - origin
        X1_dir = axes.c2p(*X[:, 1]) - origin
        span_normal = get_unit_normal(X0_dir, X1_dir)
        camera_theta = angle_of_vector(span_normal)
        camera_rotation = rotation_matrix(-camera_phi, RIGHT) @ rotation_about_z(-camera_theta - 90 * DEGREES)
        graph_group.apply_matrix(camera_rotation, about_point=ORIGIN)

        # always_redraw (rather than a one-time .next_to()) so this keeps
        # tracking y_point through any later rotations of graph_group too.
        # Kept out of graph_group itself so it stays flat/upright on screen
        # instead of inheriting graph_group's 3D tilt like y_point does.
        y_label = always_redraw(lambda: MathTex("Y", color=ORANGE).next_to(y_point, UP))

        GRID_STROKE_WIDTH = 4
        def sweep_variable(
                fixed_index, 
                fixed_value, 
                run_time=1.0, 
                color=GREEN,
                stroke_width=GRID_STROKE_WIDTH,
                lo = None,
                hi = None,
            ):
            """Fixes bhat[fixed_index] = fixed_value and animates the other
            component of bhat out to its max, then its min, then back to this
            row's baseline (fixed_index stays put, the other component
            returns to 0) -- the same increase/decrease/return-to-0 motion
            as the original single-axis "vary beta_i" animation.

            Leaves behind a static green trace of the full swept range, via
            fresh min/max trackers that get frozen (updaters cleared) once
            the sweep finishes, so repeated calls (e.g. one per grid row)
            don't contaminate each other's traces.

            Returns the trace Line.
            """
            varying_index = 1 - fixed_index
            base = np.zeros(2)
            base[fixed_index] = fixed_value
            direction = np.zeros(2)
            direction[varying_index] = 1.0

            lo_bhat, hi_bhat = bhat_extremes(axes, X, base, direction)
            if lo is not None:
                lo_bhat[varying_index] = lo
            if hi is not None:
                hi_bhat[varying_index] = hi

            running_min = ArrayValueTracker(base)
            running_max = ArrayValueTracker(base)
            running_min.add_updater(lambda m: m.set_value(np.minimum(bhat.get_value(), m.get_value())))
            running_max.add_updater(lambda m: m.set_value(np.maximum(bhat.get_value(), m.get_value())))
            self.add(running_min, running_max)

            # shade_in_3d=True is what makes ThreeDCamera depth-sort this
            # against other 3D content (y_point, span_plane, ...) by actual
            # position; plain Line/VMobjects default to False, which the
            # camera treats as "always draw on top", regardless of depth.
            trace = always_redraw(lambda: Line(
                axes.c2p(*(X @ running_min.get_value())),
                axes.c2p(*(X @ running_max.get_value())),
                color=color,
                stroke_width = stroke_width,
            ).set_shade_in_3d(True))
            # Into graph_group (not just self.add) so this moves along with
            # everything else if/when graph_group gets rotated further.
            graph_group.add(trace)

            bhat.set_value(base)
            self.play(bhat.animate.set_value(hi_bhat), run_time=run_time)
            self.play(bhat.animate.set_value(lo_bhat), run_time=run_time)
            self.play(bhat.animate.set_value(base), run_time=run_time)

            running_min.clear_updaters()
            running_max.clear_updaters()
            self.remove(running_min, running_max)

            return trace

        # A 2D scatter/trendline plot on the left, showing the actual (X1, Y)
        # data next to the abstract 3D span picture.
        X1_vals = X[:, 1]
        graph_2d = Axes(
            x_range=[-0.5, X1_vals.max() + 0.5, 1],
            y_range=[0, Y.max() + 1, 1],
            x_length=4,
            y_length=4,
            tips=False,
        )
        graph_2d.to_edge(LEFT)
        graph_2d_labels = graph_2d.get_axis_labels(x_label="X_1", y_label="Y")

        X_ticks_2d = VGroup(*[
            Line(DOWN * 0.1, UP * 0.1).move_to(graph_2d.c2p(x, 0))
            for x in X1_vals
        ])
        data_points_2d = VGroup(*[
            Dot(graph_2d.c2p(x, y), color=WHITE, radius=0.08)
            for x, y in zip(X1_vals, Y)
        ])
        trendline_2d = always_redraw(lambda: graph_2d.plot(
            lambda x: bhat.get_value()[0] + bhat.get_value()[1] * x,
            color=YELLOW,
        ))
        graph_2d_group = VGroup(graph_2d, graph_2d_labels, X_ticks_2d, data_points_2d)

        X_tex = MathTex("X = " + numpy_to_latex(X))
        Y_tex = MathTex("Y = " + numpy_to_latex(Y))
        X_tex.to_corner(UR)
        Y_tex.next_to(X_tex, DOWN)
        
        def make_bhat_tex():
            return MathTex(
                r"\hat{\beta} = " + numpy_to_latex(bhat.get_value())
            ).next_to(Y_tex, DOWN)

        # always_redraw's mob.become(...) swaps in fresh submobjects each frame,
        # which drops out of add_fixed_in_frame_mobjects's tracked set, so we
        # re-register the fixed submobjects on every refresh instead.
        bhat_tex = make_bhat_tex()

        def _refresh_bhat_tex(m):
            m.become(make_bhat_tex())
            self.add(m)

        bhat_tex.add_updater(_refresh_bhat_tex)

        with self.voiceover("X and Y are fixed at the time of data collection.") as tracker:
            # self.add_fixed_in_frame_mobjects(X_tex, Y_tex)
            self.play(FadeIn(X_tex, Y_tex, graph_2d_group))
        with self.voiceover("But beta hat can vary.") as tracker:
            # self.add_fixed_in_frame_mobjects(bhat_tex)
            self.play(FadeIn(bhat_tex))
            self.play(FadeIn(trendline_2d))
        with self.voiceover("If beta hat is the zero vector, so is y hat. Now let's look at what happens if") as tracker:
            self.add(graph_group, y_label)
        with self.voiceover("we vary beta zero hat. Y hat moves along this") as tracker:
            sweep_variable(fixed_index=1, fixed_value=0.0, lo = -1.8, hi = 1.8)
        with self.voiceover("line in the direction of (1,1,1). Now let's look at what happens if") as tracker:
            # Not shade_in_3d: these are static reference arrows, not part of
            # the Y-vs-plane depth story, and being long relative to the grid
            # they cross makes Manim's per-object (not per-pixel) depth-sort
            # z-fight around the tip. Left at the default, they just always
            # draw on top, intact.
            X0_arrow = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 0]), buff=0)
            graph_group.add(X0_arrow)
            self.play(FadeIn(X0_arrow))
        with self.voiceover("we vary beta one hat. Y hat moves along") as tracker:
            sweep_variable(fixed_index=0, fixed_value=0.0, lo = -1.8, hi = 1.8)
        with self.voiceover("in the direction of X1.") as tracker:
            X1_arrow = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 1]), buff=0)
            graph_group.add(X1_arrow)
            self.play(FadeIn(X1_arrow))
        with self.voiceover("So you can see how, by varying both beta zero hat and beta 1 hat, Y hat can be anything that's in the span of the two columns of X.") as tracker:
            # Tunable constants: how many beta1 steps to draw gridlines at,
            # and how long each row's animated sweep across beta0 takes
            # (total time is roughly GRID_ROWS * GRID_ROW_RUN_TIME * 3, since
            # each row does the full increase/decrease/return-to-0 sweep).
            GRID_ROWS = 5
            GRID_ROW_RUN_TIME = 0.3

            # Do the sweeps
            # beta1_min, beta1_max = bhat_extremes(axes, X, np.array([0., 0.]), np.array([0., 1.]))
            grid_radius = 0.9
            beta1_steps = np.linspace(-grid_radius, grid_radius, GRID_ROWS)
            for beta1 in beta1_steps:
                sweep_variable(
                    fixed_index=1, 
                    fixed_value=beta1, 
                    run_time=GRID_ROW_RUN_TIME,
                    lo = -grid_radius,
                    hi = grid_radius,
                )

            # beta0_min, beta0_max = bhat_extremes(axes, X, np.array([0., 0.]), np.array([1., 0.]))
            beta0_steps = np.linspace(-grid_radius, grid_radius, GRID_ROWS)
            for beta0 in beta0_steps:
                sweep_variable(
                    fixed_index=0, 
                    fixed_value=beta0, 
                    run_time=GRID_ROW_RUN_TIME,
                    lo = -grid_radius,
                    hi = grid_radius,
                )

            yhat_point_visible["value"] = False
            # Rather than just fading in a solid plane, sell "the span is
            # literally every one of these lines" by 1) extending the grid
            # we just swept out further along both directions, then 2)/3)
            # repeatedly doubling its density in place until the lines are
            # packed tightly enough to read as a solid surface. Each pass
            # only draws the NEW rows/columns at that resolution (the ones
            # from the previous pass are already on screen), batched into a
            # single Create so we're not replaying the slow animated
            # point-sweep for dozens of lines -- that visual already did its
            # job during the sweeps above.
            EXTENDED_RADIUS = 1.8   # how far the grid reaches after step 1
            FINER_PASSES = 4        # how many doubling-density passes (steps 2, 3, ...)
            FINER_RUN_TIME = 0.6    # time for the first doubling pass (multiplies by 0.6 each pass after)

            drawn_steps = {1: set(np.round(beta1_steps, 6)), 0: set(np.round(beta0_steps, 6))}

            def grid_lines_for_new_steps(fixed_index, all_steps, radius, stroke_width=GRID_STROKE_WIDTH, include_existing_lines = False):
                varying_index = 1 - fixed_index
                new_steps = [v for v in all_steps if include_existing_lines or round(v, 6) not in drawn_steps[fixed_index]]
                lines = VGroup()
                for val in new_steps:
                    lo, hi = np.zeros(2), np.zeros(2)
                    lo[fixed_index] = hi[fixed_index] = val
                    lo[varying_index], hi[varying_index] = -radius, radius
                    lines.add(Line(axes.c2p(*(X @ lo)), axes.c2p(*(X @ hi)), color=GREEN, stroke_width=stroke_width).set_shade_in_3d(True))
                drawn_steps[fixed_index].update(round(v, 6) for v in new_steps)
                return lines

            def add_grid_pass(radius, num_steps, run_time, stroke_width=GRID_STROKE_WIDTH, include_existing_lines = False):
                all_steps = np.linspace(-radius, radius, num_steps)
                new_lines = VGroup(
                    grid_lines_for_new_steps(1, all_steps, radius, stroke_width, include_existing_lines),
                    grid_lines_for_new_steps(0, all_steps, radius, stroke_width, include_existing_lines),
                )
                # Into graph_group (not just self.add via Create) so these
                # lines move along with everything else if/when graph_group
                # gets rotated further -- confirmed this doesn't double-add
                # or double-render, Scene.play() only auto-adds a mobject if
                # it isn't already part of an on-screen family.
                graph_group.add(new_lines)
                self.play(Create(new_lines), run_time=run_time)

            # 1) Extend: same spacing as the sweep above (not just the same
            # step *count*, which over a wider radius would coarsen it),
            # just covering more of the plane.
            original_spacing = (2 * grid_radius) / (GRID_ROWS - 1)
            extended_num_steps = int(round(2 * EXTENDED_RADIUS / original_spacing)) + 1
            add_grid_pass(EXTENDED_RADIUS, extended_num_steps, run_time=1.0, include_existing_lines=True)

            # 2)/3) Finer and finer: double the row count each pass (which
            # keeps every previous line's position and only adds the
            # in-between ones), speeding up as there's more to draw.
            num_steps = extended_num_steps
            run_time = FINER_RUN_TIME
            for _ in range(FINER_PASSES):
                num_steps = num_steps * 2 - 1
                add_grid_pass(EXTENDED_RADIUS, num_steps, run_time=run_time, stroke_width=GRID_STROKE_WIDTH, include_existing_lines=False)
                run_time *= 0.6
            self.wait()

        with self.voiceover("But out of all those possible values of Y hat, our model uses the one that minimizes the sum of squared differences between Y hat and Y.") as tracker:
            yhat_point_visible["value"] = True


            # One dashed segment + one square per data point, each square's
            # side equal to that point's |residual| (in screen distance, so
            # it reads as an actual square regardless of how graph_2d's x/y
            # axes happen to be scaled relative to each other). Wrapped in
            # always_redraw so they track bhat live, same as trendline_2d.
            def residuals_group():
                group = VGroup()
                for x, y in zip(X1_vals, Y):
                    yhat_val = bhat.get_value()[0] + bhat.get_value()[1] * x
                    p_y = graph_2d.c2p(x, y)
                    p_yhat = graph_2d.c2p(x, yhat_val)
                    dashed = DashedLine(p_y, p_yhat, color=YELLOW, stroke_width=2)
                    side = max(abs(p_y[1] - p_yhat[1]), 0.01)
                    bottom = min(p_y[1], p_yhat[1])
                    square = Square(side_length=side, color=YELLOW, fill_opacity=0.3, stroke_width=1)
                    square.move_to([p_y[0] + side / 2, bottom + side / 2, 0])
                    group.add(dashed, square)
                return group

            residuals = always_redraw(residuals_group)
            self.play(FadeIn(residuals), run_time=self.get_current_voiceover_duration() / 3)

            # The grid's viewing angle looks almost exactly down the plane's
            # normal (deliberately, so the grid itself isn't foreshortened) --
            # but that also means displacement *along* that normal, like how
            # far Y sits off the plane, projects to almost no on-screen
            # distance at all: from here on we care about exactly that
            # displacement, so keep rotating to bring it into view over time.
            #
            # graph_group.add_updater alone only rotates axes/yhat_point/
            # y_point -- anything positioned via axes.c2p() but added to the
            # *scene* directly (grid lines, arrows, ...) never actually
            # becomes part of graph_group's family, so rotating graph_group
            # doesn't touch it and it just sits frozen. Fixed at each of
            # those call sites above by adding into graph_group instead of
            # self.add()/Create()'s implicit add -- confirmed that doesn't
            # double-render, since Scene only auto-adds a mobject if it
            # isn't already part of an on-screen family.
            #
            # current_rotation tracks the actual accumulated rotation matrix
            # (updated every frame alongside the real g.rotate() call) so
            # that anything built later (e.g. the right-angle marker) can
            # still get its direction vectors exactly right, the same way
            # the one-shot version used a fixed total_rotation matrix.
            SPIN_RATE = 0.2  # radians/second
            spin_axis = UP
            current_rotation = {"matrix": camera_rotation}

            def _spin(g, dt):
                angle = SPIN_RATE * dt
                g.rotate(angle, axis=spin_axis, about_point=ORIGIN)
                current_rotation["matrix"] = rotation_matrix(angle, spin_axis) @ current_rotation["matrix"]

            graph_group.add_updater(_spin)

            # Then set Y hat to be the OLS estimate: watch the residual
            # squares shrink to their minimum total area as bhat gets there.
            self.play(bhat.animate.set_value(bhat_ols), run_time=2 * self.get_current_voiceover_duration() / 3)
        with self.voiceover("Now here's the key idea. This is the same as minimizing the Euclidean distance between Y and Y hat.") as tracker:
            # bhat is at bhat_ols now, so this is exactly the (perpendicular,
            # by construction of Y) residual vector e = Y - Y hat in R^3.
            residual_3d_line = DashedLine(axes.c2p(*Y), axes.c2p(*yhat.get_value()), color=WHITE)
            graph_group.add(residual_3d_line)
            self.play(Create(residual_3d_line), run_time=self.get_current_voiceover_duration())
        with self.voiceover("Now what point on this plane is the closest to Y? The orthogonal projection of Y onto this plane.") as tracker:
            # A small elbow marker built from raw 3D points rather than
            # manim's Angle/RightAngle: those go through line_intersection,
            # which only supports lines lying in the xy-plane (z=0) and would
            # raise on our rotated 3D lines.
            def right_angle_marker(corner, dir1, dir2, length=0.25, color=WHITE):
                u, v = normalize(dir1), normalize(dir2)
                marker = VMobject(color=color, stroke_width=3)
                marker.set_points_as_corners([
                    corner + length * u,
                    corner + length * u + length * v,
                    corner + length * v,
                ])
                return marker

            # Direction vectors for the marker come from current_rotation's
            # matrix (the live accumulated rotation, since graph_group keeps
            # spinning) applied to the *raw* data-space vectors, not from
            # axes.c2p(): ThreeDAxes' default x/y/z ranges give it different
            # unit_size per axis, so c2p() scales non-uniformly and doesn't
            # preserve angles between non-axis-aligned vectors like these --
            # only the (data-space-exact) orthogonality between X's columns
            # and the residual, carried through the angle-preserving
            # rotation, does.
            yhat_pos = axes.c2p(*yhat.get_value())
            residual_true = current_rotation["matrix"] @ (Y - X @ bhat.get_value())
            X0_true = current_rotation["matrix"] @ X[:, 0]
            right_angle = right_angle_marker(yhat_pos, X0_true, residual_true)
            graph_group.add(right_angle)
            self.play(Create(right_angle), run_time=self.get_current_voiceover_duration())
        with self.voiceover("So we want the hat matrix, when multiplied by Y, to get the orthogonal projection of Y onto this plane.") as tracker:
            ...
        with self.voiceover("That would be an orthogonal projection matrix. It should have eigenvalues of 1 for all the columns of X, and 0 for everything orthogonal to all the columns of X.") as tracker:
            ...