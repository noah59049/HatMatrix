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
        with self.voiceover("So this here is the hat matrix, which we call H.") as tracker:
            self.add(hm_formula)
            hm_formula_box = SurroundingRectangle(hm_formula, color = RED)
            self.play(Create(hm_formula_box))
            self.play(TransformByGlyphMap(hm_formula.copy(), hm_formula2,
                                          (FadeIn, [0,1])))
        with self.voiceover("Here are some facts about the hat matrix.") as tracker:
            self.play(
                FadeOut(tex7, hm_formula, hm_formula_box),
                hm_formula2.animate.to_edge(UP)
            )
        
        fact1 = MathTex(r"H^T = H")
        fact2 = MathTex(r"H^2 = H")
        fact3 = MathTex(r"H X = X")
        fact4 = MathTex(r"\vec{v} \perp X \implies H \vec{v} = 0")

        facts = VGroup(fact1, fact2, fact3, fact4).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
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


        with self.voiceover("It's symmetric. By the spectral theorem, this means it must be orthogonally diagonalizable.") as tracker:
            self.play(LaggedStart(Create(checks[0]), Write(fact1), lag_ratio=0.6))
            proof11 = MathTex("H^T = (X (X^T X)^{-1} X^T)^T")
            proof12 = MathTex("H^T = X^T^T (X^T X)^{-1}^T X^T")
            proof13 = MathTex("H^T = X (X^T X)^T^{-1} X^T")
            proof14 = MathTex("H^T = X (X^T X^T^T)^{-1} X^T")
            proof15 = MathTex("H^T = X (X^T X)^{-1} X^T")
            proof16 = MathTex("H^T = H")
        with self.voiceover("It's idempotent, meaning that H squared equals H.") as tracker:
            self.play(LaggedStart(Create(checks[1]), Write(fact2), lag_ratio=0.6))
            proof21 = MathTex("H^2 = X (X^T X)^{-1} X^T X (X^T X)^{-1} X^T")
            proof22 = MathTex("H^2 = X (X^T X)^{-1} X^T")
            proof23 = MathTex("H^2 = H")
        with self.voiceover("H times X is X. Since H leaves the entire matrix X unchanged, it must leave each column of X unchanged.") as tracker:
            self.play(LaggedStart(Create(checks[2]), Write(fact3), lag_ratio=0.6))
            proof31 = MathTex("H X = X (X^T X)^{-1} X^T X")
            proof32 = MathTex("H X = X")
        with self.voiceover("If you take H times any vector v orthogonal to X, this X transpose v makes the whole product turn out to zero.") as tracker:
            self.play(LaggedStart(Create(checks[3]), Write(fact4), lag_ratio=0.6))
            proof41 = MathTex(r"H \vec{v} = X (X^T X)^{-1} X^T \vec{v}")
            proof42 = MathTex(r"H \vec{v} = X (X^T X)^{-1} 0")
            proof43 = MathTex(r"H \vec{v} = 0")
        with self.voiceover("These last 2 facts are an intuition for how H is a projection onto the column space of X.") as tracker:
            ...