import numpy as np
from manim import *
from N_Tools import animate_matrix_vector_product, numpy_to_latex


class TestMatMul(Scene):
    def construct(self):
        M = MathTex("X = " + numpy_to_latex(np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])))
        v = MathTex(r"\hat{\beta} = " + numpy_to_latex(np.array([1.5, -2.0, 0.5])))
        M.move_to(LEFT * 3 + UP * 1.5)
        v.next_to(M, DOWN, aligned_edge=LEFT, buff=0.6)

        self.add(M, v)
        self.wait(0.3)
        anim = animate_matrix_vector_product(M[0], v)
        self.play(anim, run_time=3)
        self.wait(0.3)
