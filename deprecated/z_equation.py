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
        
        vertex_forms = [
            rf"y=a(x-h)^2+k",
            rf"H_ii=a(x-h)^2+k",
            rf"H_ii=a(x-\mu_X)^2+k",
            rf"H_ii=a(x-\mu_X)^2+{1/n:.3f}",
            rf"H_ii=a(x-\mu_X)^2+{1/n:.3f}",
        ]

        vertex_form_texes_list = [MathTex(v).to_edge(DOWN) for v in vertex_forms]
        vertex_form_texes = VGroup(*vertex_form_texes_list)


