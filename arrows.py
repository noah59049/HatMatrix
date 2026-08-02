from manim import *

class ArrowScene(ThreeDScene):
    def construct(self):
        X = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0],
        ])
        bhat_ols = np.array([0.3, 0.7])
        Y_residual_scale = 3  # how far off the X0/X1 plane Y sits
        yhat = X @ bhat_ols
        Y = yhat + Y_residual_scale * get_unit_normal(X[:, 0], X[:, 1])

        axes = ThreeDAxes().to_corner(UL)
        origin_pos = axes.c2p(0, 0, 0)
        Y_pos = axes.c2p(*Y)
        yhat_pos = axes.c2p(*yhat)

        arrow_to_Y = Arrow(origin_pos, Y_pos, buff=0, color=ORANGE)
        arrow_to_yhat = Arrow(origin_pos, yhat_pos, buff=0, color=YELLOW)
        arrow_yhat_to_Y = Arrow(yhat_pos, Y_pos, buff=0, color=WHITE)        

        graph_group = VGroup(axes, arrow_to_Y, arrow_to_yhat, arrow_yhat_to_Y)

        
        camera_phi = 70 * DEGREES
        origin = axes.c2p(0, 0, 0)
        X0_dir = axes.c2p(*X[:, 0]) - origin
        X1_dir = axes.c2p(*X[:, 1]) - origin
        span_normal = get_unit_normal(X0_dir, X1_dir)
        camera_theta = angle_of_vector(span_normal)
        camera_rotation = rotation_matrix(-camera_phi, RIGHT) @ rotation_about_z(-camera_theta - 90 * DEGREES)
        graph_group.apply_matrix(camera_rotation, about_point=ORIGIN)

        eq_Y = arrow_to_Y.copy()
        eq_yhat = arrow_to_yhat.copy()
        eq_e = arrow_yhat_to_Y.copy()
        equation = VGroup(
            eq_Y, MathTex("="), eq_yhat, MathTex("+"), eq_e
        ).arrange(RIGHT, buff=0.4).move_to(ORIGIN)

        self.play(FadeIn(graph_group))
        
        self.play(
            ReplacementTransform(arrow_to_Y, equation[0]),
            Write(equation[1]),
            ReplacementTransform(arrow_to_yhat, equation[2]),
            Write(equation[3]),
            ReplacementTransform(arrow_yhat_to_Y, equation[4]),
        )
        self.wait(1)
