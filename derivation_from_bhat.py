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
        tex6 = MathTex(r"\hat{\beta} = (X^T X)^{-1} X^T Y")

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
        with self.voiceover("left multiply by the inverse of X transpose X, and then we get our formula for beta hat. It's kind of amazing it has a closed form solution.") as tracker:
            self.play(TransformByGlyphMap(tex3, tex4,
                                          (FadeIn, range(0,7)),
                                          (FadeIn, range(11,18), {"run_time": 0.45, "delay":0.5},),
                                    ))
            self.play(TransformByGlyphMap(tex4, tex5,
                                          (range(11,21), FadeOut, {"run_time":0.5})))