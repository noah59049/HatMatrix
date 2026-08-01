import numpy as np
from manim import *
from N_Tools import *
from stitcher_scene import StitcherScene


class XSpan(StitcherScene, ThreeDScene):
    def construct_scene(self):
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
        bhat = ArrayValueTracker([np.nan, np.nan])

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
        # shade_in_3d=False (Dot3D's family defaults to True): yhat_point's
        # position is axes.c2p(X @ bhat), and every grid line is also built
        # from X @ (something), so yhat_point is *exactly* coplanar with the
        # entire grid at all times -- not an approximation, a structural
        # fact. Depth-sorting a point against a plane it's always sitting
        # exactly on is degenerate; with real depth-sort enabled it loses
        # essentially every near-tie against the ~100+ grid lines (added to
        # graph_group's family well after it, so stable-sort ties resolve
        # against it) and renders permanently buried underneath. It never
        # needed real depth-compositing here -- like the arrows, it should
        # just always draw on top.
        yhat_point = always_redraw(lambda: Dot3D(
            axes.c2p(*yhat.get_value()), color=YELLOW, radius=0.1
        ).set_opacity(1.0 if yhat_point_visible["value"] else 0.0).set_shade_in_3d(False))
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

        # Raw-data table (X1, Y side by side) and the intermediate "X as a
        # plain vector, before the ones column" step -- both only exist to
        # be morphed into Y_tex/X_tex during the intro, so their starting
        # position doesn't matter; Transform animates them to wherever the
        # target already is.
        #
        # Built from MathTex + numpy_to_latex (the same pipeline X_tex/
        # Y_tex/bhat_tex use) rather than a Table mobject: guarantees
        # identical font size and spacing automatically, no scaling needed
        # to make it match them.
        X1_col_tex = Tex(numpy_to_latex(X1_vals, make_table = True))
        Y_col_tex = Tex(numpy_to_latex(Y, make_table = True))
        x_col_group = VGroup(MathTex("X"), X1_col_tex).arrange(DOWN, buff=0.3)
        y_col_group = VGroup(MathTex("Y"), Y_col_tex).arrange(DOWN, buff=0.3)
        data_table = VGroup(x_col_group, y_col_group).arrange(RIGHT, buff=0.0).to_corner(UR)

        def get_bhat_string():
            text = numpy_to_latex(bhat.get_value())
            if np.isnan(bhat.get_value()[0]):
                text = text.replace("nan", r"\hat{\beta}_0", count = 1)
            if np.isnan(bhat.get_value()[1]):
                text = text.replace("nan", r"\hat{\beta}_1", count = 1)
            return text

        def make_bhat_tex():
            return MathTex(r"\hat{\beta} = " + get_bhat_string()).next_to(Y_tex, DOWN)

        bhat_tex = always_redraw(make_bhat_tex)

        with self.voiceover("Let's start by looking at a very simple example, linear regression with just 1 independent variable and 3 data points.") as tracker:
            pass
        with self.voiceover("Let's show the scatterplot") as tracker:
            self.play(FadeIn(graph_2d_group))

        with self.voiceover("and a data table. Right now we have our Ys as separate quantities, but") as tracker:
            self.play(FadeIn(data_table))

        with self.voiceover("it's useful to think of them as a vector, a column vector, for reasons that will become apparent.") as tracker:
            y_column = y_col_group.copy()
            self.play(TransformMatchingShapes(y_column, Y_tex))

        with self.voiceover("Beta hat is also a column vector.") as tracker:
            self.play(FadeIn(bhat_tex))

        with self.voiceover("X, on the other hand, is represented as a matrix. Each column has one of our X variables, and on the very left there's this column of ones.") as tracker:
            X1_vector_tex = MathTex("X = " + numpy_to_latex(X1_vals)).to_corner(UR)
            x_column = x_col_group.copy()
            self.play(FadeOut(data_table, run_time = 0.5), TransformMatchingShapes(x_column, X1_vector_tex))
            self.play(TransformMatchingShapes(X1_vector_tex, X_tex))

        with self.voiceover("It's defined like this because X beta hat is equal to Y hat this way, where Y hat is a vector in the same manner as Y.") as tracker:
            xbhat_tex = MathTex(
                r"X",
                r"\hat{\beta}",
                r"=",
                numpy_to_latex(X),
                get_bhat_string()
            ).move_to(RIGHT * 0.64)
            self.play(
                TransformFromCopy(X_tex[0][0], xbhat_tex[0]),
                TransformFromCopy(bhat_tex[0][0:2], xbhat_tex[1]),
                TransformFromCopy(bhat_tex[0][2], xbhat_tex[2]),
                TransformFromCopy(X_tex[0][2:], xbhat_tex[3]),
                TransformFromCopy(bhat_tex[0][3:], xbhat_tex[4]),
            )
            
        with self.voiceover("and I'll leave up an animation for the intuition and a more rigorous proof onscreen.") as tracker:
            matmul_animation = animate_matrix_vector_product(
                xbhat_tex[3], 
                xbhat_tex[4], 
                buff=0.6,
                place = lambda x : x.next_to(xbhat_tex, DOWN)
            )

            self.play(matmul_animation)
            lower_down_equals = MathTex("=").next_to(matmul_animation.mobB, LEFT)
            self.add(lower_down_equals)
            # TODO: Put the more rigorous proof onscreen too

        return

        with self.voiceover("Let's show how Y hat depends on beta hat, starting with if beta hat is zero. Then all the Y hats are zero,") as tracker:
            self.play(FadeOut(matmul_animation.mobB, lower_down_equals, xbhat_tex))
            bhat.set_value(np.array([0.0,0.0]))
            self.play(FadeIn(trendline_2d))
            graph_2d_group.add(trendline_2d)
        with self.voiceover("or in other words, the y hat vector is zero.") as tracker:
            self.add(graph_group, y_label)
        with self.voiceover("Now if we vary beta 0 hat, all of the Y hats increase by the same amount. So representing Y hat as a vector,") as tracker:
            sweep_variable(fixed_index=1, fixed_value=0.0, lo = -1.8, hi = 1.8)
        with self.voiceover("that's moving along (1,1,1).") as tracker:
            # Not shade_in_3d: these are static reference arrows, not part of
            # the Y-vs-plane depth story, and being long relative to the grid
            # they cross makes Manim's per-object (not per-pixel) depth-sort
            # z-fight around the tip. Left at the default, they just always
            # draw on top, intact.
            X0_arrow = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 0]), buff=0)
            graph_group.add(X0_arrow)
            self.play(FadeIn(X0_arrow))
        with self.voiceover("Now if we vary beta one hat, the Y hat that's at X=0 doesn't change at all, and the Y hat that's at X = 1 changes a little bit, and the Y hat that's at X = 3 changes 3 times as much.") as tracker:
            sweep_variable(fixed_index=0, fixed_value=0.0, lo = -1.8, hi = 1.8)
        with self.voiceover("On the other graph, Y hat moves in the direction of (0,1,3), or X1.") as tracker:
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
            graph_2d_group.add(residuals)

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
            residual_3d_line = always_redraw(lambda: DashedLine(axes.c2p(*Y), axes.c2p(*yhat.get_value()), color=WHITE))
            graph_group.add(residual_3d_line)
            self.add(residual_3d_line)
        with self.voiceover("And the length of the line between Y and Y hat is minimized precisely when that line is perpendicular to the span of X.") as tracker:
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
            def make_right_angle():
                yhat_pos = axes.c2p(*yhat.get_value())
                X0_true = current_rotation["matrix"] @ X[:, 0]
                residual_true = current_rotation["matrix"] @ (Y - X @ bhat.get_value())
                return right_angle_marker(yhat_pos, X0_true, residual_true)
            right_angle = always_redraw(make_right_angle)
            graph_group.add(right_angle)
            # FadeIn, not Create -- same reason as residual_3d_line above.
            self.add(right_angle)

        with self.voiceover("In linear algebra jargon, this is called an orthogonal projection.") as tracker:
            label = Text("Orthogonal Projection").to_corner(UL)
            self.play(FlashOn(label, run_time = [0.4, self.get_current_voiceover_duration() - 1, 0.4]))
        with self.voiceover("Y hat is the orthogonal projection of Y onto the column space of X. Therefore, the hat matrix is going to be an orthogonal projection matrix onto the column space of X.") as tracker:
            # A 3D lattice of unit-spaced points/lines "scaffolding" ambient
            # space, then collapsed onto col(X) via the actual orthogonal
            # projection matrix H = X(X'X)^-1 X' -- this *is* the hat
            # matrix, just not named as such until the next line.
            H = X @ np.linalg.inv(X.T @ X) @ X.T
            LATTICE_SIZE = 3  # points per axis (odd, so it centers on the origin); e.g. 5 -> a 5x5x5 lattice
            # Keep at 3 for testing, increase to 7 or more for production

            steps = list(range(-(LATTICE_SIZE // 2), LATTICE_SIZE // 2 + 1))

            # alpha=0 -> original lattice position, alpha=1 -> its orthogonal
            # projection onto col(X). Shared across the whole scaffold so one
            # animation moves everything at once. Interpolating in data space
            # and only converting via axes.c2p() inside always_redraw (rather
            # than animating .move_to()/.put_start_and_end_on() targets
            # computed once up front) is deliberate: graph_group is still
            # continuously rotating (the _spin updater above), and a
            # snapshot-based position animation would fight that the same
            # way Create fought always_redraw earlier in this file -- it'd
            # freeze these points relative to wherever the rotation was when
            # the animation began, leaving them visibly detached from the
            # rest of the (still-rotating) scene by the time it finishes.
            project_alpha = ValueTracker(0.0)

            def lattice_pos(alpha, original):
                return (1 - alpha) * original + alpha * (H @ original)

            scaffold_dots = VGroup(*[
                always_redraw(lambda p=np.array([i, j, k], dtype=float): Dot3D(
                    axes.c2p(*lattice_pos(project_alpha.get_value(), p)), color=BLUE, radius=0.04
                ).set_shade_in_3d(True))
                for i in steps for j in steps for k in steps
            ])

            def scaffold_line(fixed_axes, fixed_vals):
                def build():
                    lo, hi = np.zeros(3), np.zeros(3)
                    for axis, val in zip(fixed_axes, fixed_vals):
                        lo[axis] = hi[axis] = val
                    varying_axis = ({0, 1, 2} - set(fixed_axes)).pop()
                    lo[varying_axis], hi[varying_axis] = steps[0], steps[-1]
                    alpha = project_alpha.get_value()
                    return Line(
                        axes.c2p(*lattice_pos(alpha, lo)),
                        axes.c2p(*lattice_pos(alpha, hi)),
                        color=BLUE, stroke_width=2,
                    ).set_shade_in_3d(True)
                return always_redraw(build)

            scaffold_lines = VGroup(*[
                scaffold_line(fixed_axes, (i, j))
                for i in steps for j in steps
                for fixed_axes in [(1, 2), (0, 2), (0, 1)]
            ])

            scaffold = VGroup(scaffold_lines, scaffold_dots)
            graph_group.add(scaffold)

            half_time = self.get_current_voiceover_duration() / 2
            self.play(FadeIn(scaffold), run_time=half_time)
            self.play(project_alpha.animate.set_value(1.0), run_time=half_time)

        orth_fact = Tex(r"$e \perp$ all cols of $X$").to_corner(UR)
        hm_derivations = [
            MathTex(r"Y =   \hat{Y} + e"),
            MathTex(r"Y = X \hat{\beta} + e"),
            MathTex(r"X^T Y = X^T X \hat{\beta} + X^T e"),
            MathTex(r"X^T Y = X^T X \hat{\beta}"),
            MathTex(r"(X^T X)^{-1} X^T Y = (X^T X)^{-1} X^T X \hat{\beta}"),
            MathTex(r"(X^T X)^{-1} X^T Y = \hat{\beta}"),
            MathTex(r"X (X^T X)^{-1} X^T Y = X \hat{\beta}"),
            MathTex(r"X (X^T X)^{-1} X^T Y = \hat{Y}"),
        ]
        hm_formula  = MathTex(r"     X (X^T X)^{-1} X^T").move_to(hm_derivations[-1], aligned_edge=LEFT)
        hm_formula2 = MathTex(r" H = X (X^T X)^{-1} X^T").next_to(hm_formula, DOWN, aligned_edge = RIGHT)

        with self.voiceover("If we have more datapoints, and possibly more variables, the graph of Y and Y hat will be more than 3 dimensions, so we can't really visualize it. But the same idea applies. Y hat can be anything in the column space of X. It's going to be the vector that's closest to Y, which will be the orthogonal projection of Y onto that column space.") as tracker:
            ...
        with self.voiceover("So we know what the hat matrix is supposed to do, but now we're going to focus on finding a formula for it.") as tracker:
            pass
        with self.voiceover("For this it's going to be useful to decompose Y into a component that's in the column space of X, and a component that's orthogonal to the column space of X.") as tracker:
            origin_pos = axes.c2p(0, 0, 0)
            Y_pos = axes.c2p(*Y)
            yhat_pos = axes.c2p(*yhat.get_value())

            arrow_to_Y = Arrow(origin_pos, Y_pos, buff=0, color=ORANGE)
            arrow_to_yhat = Arrow(origin_pos, yhat_pos, buff=0, color=YELLOW)
            arrow_yhat_to_Y = Arrow(yhat_pos, Y_pos, buff=0, color=WHITE)
            self.play(GrowArrow(arrow_to_Y), GrowArrow(arrow_to_yhat), GrowArrow(arrow_yhat_to_Y))

            # Fade out everything except those 3 arrows -- we're leaving the
            # rotating 3D visualization behind for the symbolic derivation
            # below.
            scene_to_fade = VGroup(graph_group, graph_2d_group, X_tex, Y_tex, bhat_tex, y_label)
            # A lot of this (the graph_group spin updater, yhat_point, the
            # sweep traces, the scaffold lattice, residual_3d_line,
            # right_angle, trendline_2d, y_label, bhat_tex) is always_redraw
            # or has its own per-frame updater, which rebuilds each of them
            # fresh -- at full opacity -- every single frame. FadeOut
            # animates opacity down, but those updaters would keep
            # overwriting it back to full each frame, fighting the fade
            # exactly like Create fought always_redraw earlier in this
            # file. clear_updaters(recursive=True, the default) stops every
            # updater in the whole family tree first (graph_group's own
            # _spin included), so FadeOut's opacity change actually sticks.
            scene_to_fade.clear_updaters()
            assert len(bhat_tex.updaters) == 0
            self.play(FadeOut(scene_to_fade))

            # {arrow to Y} = {arrow to Y hat} + {arrow from Y hat to Y}
            eq_Y = arrow_to_Y.copy()
            eq_yhat = arrow_to_yhat.copy()
            eq_e = arrow_yhat_to_Y.copy()
            equation = VGroup(
                eq_Y, MathTex("="), eq_yhat, MathTex("+"), eq_e
            ).arrange(RIGHT, buff=0.4).move_to(ORIGIN)
            self.play(
                ReplacementTransform(arrow_to_Y, eq_Y),
                ReplacementTransform(arrow_to_yhat, eq_yhat),
                ReplacementTransform(arrow_yhat_to_Y, eq_e),
                Write(equation[1]),
                Write(equation[3]),
            )
        with self.voiceover("Luckily for us, these vectors have names: The component in the column space is Y hat, and the component orthogonal to the column space is e, our residual vector. Now we can simplify. Since X beta hat is") as tracker:
            equation_tex = VGroup(
                MathTex("Y"), MathTex("="), MathTex(r"\hat{Y}"), MathTex("+"), MathTex("e")
            ).arrange(RIGHT, buff=0.4).move_to(ORIGIN)
            self.play(TransformIndices(equation, equation_tex))
            self.play(
                TransformMatchingShapes(equation_tex, hm_derivations[0]),
                FlashOn(orth_fact, run_time = (0.4, self.get_current_voiceover_duration() - 2, 0.4)),
            )
        with self.voiceover("equal to Y hat, we'll substitute that in. Then we") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[0], hm_derivations[1],
                                          ([3],[3,4]),
                                          ))
        with self.voiceover("left multiply by X transpose. Since e is orthogonal to every column of X,") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[1], hm_derivations[2],
                                          (FadeIn, [0,1]),
                                          (FadeIn, [4,5]),
                                          (FadeIn, [10,11]),
                                          ))
        with self.voiceover("X transpose e is zero. Then we") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[2], hm_derivations[3],
                                          ([9,10,11,12], FadeOut, {"run_time": 0.5})))
        with self.voiceover("left multiply by the inverse of X transpose X,") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[3], hm_derivations[4],
                                          (FadeIn, range(0,7)),
                                          (FadeIn, range(11,18), {"run_time": 0.45, "delay":0.5},),
                                    ))
        with self.voiceover("Cancel terms, and then we get our formula for beta hat.") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[4], hm_derivations[5],
                                          (range(11,21), FadeOut, {"run_time":0.5})))
        
        with self.voiceover("Now we can left multiply by X") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[5], hm_derivations[6],
                                          (FadeIn, [0]),
                                          (FadeIn, [12]),
                                          ))
        with self.voiceover("and X beta hat is Y hat.") as tracker:
            self.play(TransformByGlyphMap(hm_derivations[6], hm_derivations[7],
                                          ([12,14], [13])))
        with self.voiceover("So this here is the hat matrix, which we call H.") as tracker:
            self.add(hm_formula)
            hm_formula_box = SurroundingRectangle(hm_formula, color = RED)
            self.play(Create(hm_formula_box))
            self.play(TransformByGlyphMap(hm_formula.copy(), hm_formula2,
                                          (FadeIn, [0,1])))
        with self.voiceover("Here are some facts about the hat matrix. I'll leave up proofs on screen if you're curious, but the most important thing about these facts is how they relate to each other and to what the hat matrix is supposed to do.") as tracker:
            self.play(
                FadeOut(hm_derivations[7], hm_formula, hm_formula_box),
                hm_formula2.animate.to_edge(UP)
            )
        
        fact1 = MathTex(r"H^T = H")
        fact2 = MathTex(r"H^2 = H")
        fact3 = MathTex(r"H X = X")
        fact4 = MathTex(r"\vec{v} \perp X \implies H \vec{v} = 0")
        fact5 = MathTex(r"\lambda \in \{0,1\}.")

        facts = VGroup(fact1, fact2, fact3, fact4, fact5).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        facts.next_to(hm_formula2, DOWN, buff=1.0)

        def make_checkmark():
            checkmark = VMobject(stroke_color=GREEN, stroke_width=7)
            checkmark.set_points_as_corners([
                [-0.18, 0.02, 0],
                [-0.04, -0.16, 0],
                [0.22, 0.2, 0],
            ])
            return checkmark

        checks = VGroup(*(make_checkmark().next_to(fact, LEFT, buff=0.35) for fact in facts))


        def make_proof_group(fact, *steps):
            label = Tex("Proof:")
            step_group = VGroup(*steps).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            group = VGroup(label, step_group).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            group.scale(0.55)
            step_group.shift(RIGHT * 0.4)
            group.next_to(fact, DOWN, aligned_edge=LEFT, buff=0.3).shift(RIGHT * 0.5)
            return group

        with self.voiceover("It's symmetric. By the spectral theorem, this means it must be orthogonally diagonalizable.") as tracker:
            self.play(LaggedStart(Create(checks[0]), Write(fact1), lag_ratio=0.6))
            proof11 = MathTex("H^T = (X (X^T X)^{-1} X^T)^T")
            proof12 = MathTex("H^T = {X^T}^T {(X^T X)^{-1}}^T X^T")
            proof13 = MathTex("H^T = X {(X^T X)^T}^{-1} X^T")
            proof14 = MathTex("H^T = X (X^T {X^T}^T)^{-1} X^T")
            proof15 = MathTex("H^T = X (X^T X)^{-1} X^T")
            proof16 = MathTex("H^T = H")
            proof1_group = make_proof_group(fact1, proof11, proof12, proof13, proof14, proof15, proof16)
            self.play(FadeIn(proof1_group))
            fact11 = Tex(r"$\implies H$ is orthogonally diagonalizable.").next_to(fact1, RIGHT)
            self.play(FadeIn(fact11))
        with self.voiceover("It's idempotent, meaning that H squared equals H.") as tracker:
            self.play(FadeOut(proof1_group))
            self.play(LaggedStart(Create(checks[1]), Write(fact2), lag_ratio=0.6))
            proof21 = MathTex("H^2 = X (X^T X)^{-1} X^T X (X^T X)^{-1} X^T")
            proof22 = MathTex("H^2 = X (X^T X)^{-1} X^T")
            proof23 = MathTex("H^2 = H")
            proof2_group = make_proof_group(fact2, proof21, proof22, proof23)
            self.play(FadeIn(proof2_group))
        with self.voiceover("H times X is X. Since H leaves the entire matrix X unchanged, it must leave each column of X unchanged.") as tracker:
            self.play(FadeOut(proof2_group))
            self.play(LaggedStart(Create(checks[2]), Write(fact3), lag_ratio=0.6))
            proof31 = MathTex("H X = X (X^T X)^{-1} X^T X")
            proof32 = MathTex("H X = X")
            proof3_group = make_proof_group(fact3, proof31, proof32)
            self.play(FadeIn(proof3_group))
            fact31 = MathTex(r"\implies H X_{\cdot j} = X_{\cdot j}").next_to(fact3, RIGHT)
            self.play(FadeIn(fact31))
        with self.voiceover("H times any vector orthogonal to X is zero.") as tracker:
            self.play(FadeOut(proof3_group))
            self.play(LaggedStart(Create(checks[3]), Write(fact4), lag_ratio=0.6))
            proof41 = MathTex(r"H \vec{v} = X (X^T X)^{-1} X^T \vec{v}")
            proof42 = MathTex(r"H \vec{v} = X (X^T X)^{-1} 0")
            proof43 = MathTex(r"H \vec{v} = 0")
            proof4_group = make_proof_group(fact4, proof41, proof42, proof43)
            self.play(FadeIn(proof4_group))
        with self.voiceover("All the eigenvalues of the hat matrix are 0 or 1, and in fact, our previous facts have been hinting at where the eigenvalues of 0 and 1 are.") as tracker:
            self.play(FadeOut(proof4_group))
            self.play(LaggedStart(Create(checks[4]), Write(fact5), lag_ratio=0.6))
            proof51 = MathTex(r"H^2=H, H\vec{v}=\lambda\vec{v}")
            proof52 = MathTex(r"H^2\vec{v}=H\vec{v}")
            proof53 = MathTex(r"\lambda^2\vec{v}=\lambda\vec{v}")
            proof54 = MathTex(r"\lambda^2=\lambda")
            proof55 = MathTex(r"\lambda \in \{0,1\}")
            proof5_group = make_proof_group(fact5, proof51, proof52, proof53, proof54, proof55)
            self.play(FadeIn(proof5_group))
        with self.voiceover("The eigenvectors with eigenvalue 1 are the columns of X and anything in their span.") as tracker:
            self.play(FlashOn(SurroundingRectangle(fact3, color = RED)))
        with self.voiceover("The eigenvectors with eigenvalue 0 are everything orthogonal to X.") as tracker:
            self.play(FlashOn(SurroundingRectangle(fact4, color = RED)))
        with self.voiceover("And the eigenspace of 0 is orthogonal to the eigenspace of 1, which is consistent with the hat matrix being orthogonally diagonalizable.") as tracker:
            self.play(FlashOn(SurroundingRectangle(fact1, color = RED)))