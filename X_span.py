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
        Y = np.array([1.0, 3.0, 4.0])

        # Setup the 3D axes with the span of X
        bhat = ArrayValueTracker([0.0, 0.0])
        
        yhat = ArrayValueTracker(X @ bhat.get_value())
        yhat.add_updater(lambda m: m.set_value(X @ bhat.get_value()))
        
        bhat_min = ArrayValueTracker(bhat.get_value())
        bhat_min.add_updater(lambda m : m.set_value(np.minimum(bhat.get_value(), bhat_min.get_value())))
        bhat_max = ArrayValueTracker(bhat.get_value())
        bhat_max.add_updater(lambda m : m.set_value(np.maximum(bhat.get_value(), bhat_max.get_value())))

        self.add(bhat, bhat_min, bhat_max, yhat)

        axes = ThreeDAxes()
        point = always_redraw(lambda: Dot3D(axes.c2p(*yhat.get_value()), color=YELLOW, radius=0.1))
        y_point = Dot3D(axes.c2p(*Y), color=WHITE, radius=0.1)
        y_label = MathTex("Y").next_to(y_point, UP)
        graph_group = VGroup(axes, point, y_point, y_label)
        
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
            self.add(graph_group)
        with self.voiceover("we vary beta zero hat. Y hat moves along this") as tracker:
            X0_trace = always_redraw(
                lambda : Line(
                    axes.c2p(X[:, 0] * bhat_min.get_value()[0]), 
                    axes.c2p(X[:, 0] * bhat_max.get_value()[0]), 
                    color = GREEN
                )
            )
            self.add(X0_trace)
            self.play(bhat.animate.set_value(np.array([ 1, 0])))
            self.play(bhat.animate.set_value(np.array([-1, 0])))
            self.play(bhat.animate.set_value(np.array([ 0, 0])))
        with self.voiceover("line in the direction of (1,1,1). Now let's look at what happens if") as tracker:
            X0_arr = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 0]), buff=0)
            self.play(FadeIn(X0_arr))
        with self.voiceover("we vary beta one hat. Y hat moves along") as tracker:
            X1_trace = always_redraw(
                lambda : Line(
                    axes.c2p(X[:, 1] * bhat_min.get_value()[1]), 
                    axes.c2p(X[:, 1] * bhat_max.get_value()[1]), 
                    color = GREEN
                )
            )
            self.add(X1_trace)
            self.play(bhat.animate.set_value(np.array([0, 1])))
            self.play(bhat.animate.set_value(np.array([0,-1])))
            self.play(bhat.animate.set_value(np.array([0, 0])))
        with self.voiceover("in the direction of X1.") as tracker:
            X1_arr = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 1]), buff=0)
            self.play(FadeIn(X1_arr))
        with self.voiceover("So you can see how, by varying both beta zero hat and beta 1 hat, Y hat can be anything that's in the span of the two columns of X.") as tracker:
            span_plane = Surface(
                lambda u, v: axes.c2p(*(u * X[:, 0] + v * X[:, 1])),
                u_range=[-1.5, 1.5],
                v_range=[-1.5, 1.5],
                resolution=(8, 8),
                fill_color=GREEN,
                fill_opacity=0.3,
                checkerboard_colors=[GREEN, GREEN],
                stroke_width=0,
            )
            self.play(FadeIn(span_plane))
        return
        with self.voiceover("But out of all those possible values of Y hat, our model only uses one. Specifically, it uses the one that minimizes the sum of squared differences between Y hat and Y.") as tracker:
            ...
        with self.voiceover("Now here's the key idea. This is the same as minimizing the Euclidean distance between Y and Y hat.") as tracker:
            ...
        with self.voiceover("Now what point on this plane is the closest to Y? The orthogonal projection of Y onto this plane.") as tracker:
            ...
        with self.voiceover("So we want the hat matrix, when multiplied by Y, to get the orthogonal projection of Y onto this plane.") as tracker:
            ...
        with self.voiceover("That would be an orthogonal projection matrix. It should have eigenvalues of 1 for all the columns of X, and 0 for everything orthogonal to all the columns of X.") as tracker:
            ...