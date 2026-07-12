import numpy as np
from manim import *
from MF_Tools import *
from stitcher_scene import StitcherScene


class DerivationFromBhat(StitcherScene):
    def construct_scene(self):
        self.silent = True
        tex1 = MathTex(r"Y = X \hat{\beta} + e")
        tex2 = MathTex(r"X^T Y = X^T X \hat{\beta} + X^T e")
        tex3 = MathTex(r"X^T Y = X^T X \hat{\beta}")
        tex4 = MathTex(r"(X^T X)^{-1} X^T Y = (X^T X)^{-1} X^T X \hat{\beta}")
        tex5 = MathTex(r"(X^T X)^{-1} X^T Y = \hat{\beta}")
        tex6 = MathTex(r"X (X^T X)^{-1} X^T Y = X \hat{\beta}")
        tex7 = MathTex(r"X (X^T X)^{-1} X^T Y = \hat{Y}")
        hm_formula  = MathTex(r"     X (X^T X)^{-1} X^T").move_to(tex7, aligned_edge=LEFT)
        hm_formula2 = MathTex(r" H = X (X^T X)^{-1} X^T").next_to(hm_formula, DOWN, aligned_edge = RIGHT)

        with self.voiceover("So we can represent as X beta hat plus e. e is our error vector, and it is orthogonal to all the columns of X. We want to isolate beta hat. The first step in here is to") as tracker:
            self.play(Write(tex1))
        with self.voiceover("left multiply by X transpose. Since e is orthogonal to every column of X,") as tracker:
            self.play(TransformByGlyphMap(tex1, tex2,
                                          (FadeIn, [0,1]),
                                          (FadeIn, [4,5]),
                                          (FadeIn, [10,11]),
                                          ))
        with self.voiceover("X transpose e is zero. Then we") as tracker:
            self.play(TransformByGlyphMap(tex2, tex3,
                                          ([9,10,11,12], FadeOut, {"run_time": 0.5})))
        with self.voiceover("left multiply by the inverse of X transpose X, and then we get our formula for beta hat.") as tracker:
            self.play(TransformByGlyphMap(tex3, tex4,
                                          (FadeIn, range(0,7)),
                                          (FadeIn, range(11,18), {"run_time": 0.45, "delay":0.5},),
                                    ))
            self.play(TransformByGlyphMap(tex4, tex5,
                                          (range(11,21), FadeOut, {"run_time":0.5})))
        
        with self.voiceover("Now we can left multiply by X") as tracker:
            self.play(TransformByGlyphMap(tex5, tex6,
                                          (FadeIn, [0]),
                                          (FadeIn, [12]),
                                          ))
        with self.voiceover("and X beta hat is Y hat.") as tracker:
            self.play(TransformByGlyphMap(tex6, tex7,
                                          ([12,13], [12])))
        with self.voiceover("So this here is the hat matrix.") as tracker:
            self.add(hm_formula)
            hm_formula_box = SurroundingRectangle(hm_formula)
            self.play(Create(hm_formula_box))
            self.play(TransformByGlyphMap(hm_formula.copy(), hm_formula2,
                                          (FadeIn, [0,1])))