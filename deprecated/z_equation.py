from manim import *
from MF_Tools import *

class ZScene(Scene):
    def construct(self):
        n = 12
        vertex_tex = Tex(rf"Vertex: $({1/n:.3f}, \mu_X)$")
        self.add(vertex_tex)
        var_a_tex = MathTex(rf"\sigma^2 a = {1/n:.3f}").next_to(vertex_tex, DOWN)
        self.add(var_a_tex)
        a_var_tex = MathTex(rf"a = \frac{{{1/n:.3f}}}{{\sigma^2}}").next_to(vertex_tex, DOWN)
        self.play(TransformByGlyphMap(var_a_tex, a_var_tex,
                                      ([], [7]),
                                      ([0,1], [8,9])))
