from manim import *
from MF_Tools import *
from N_Tools import *

class MyScene(Scene):
    def construct(self):
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

        # self.play(TransformWithBoxes(hm_derivations[0], hm_derivations[1],
        #                                 ([3],[3,4]),
        #                                 ))

        # self.play(TransformWithBoxes(hm_derivations[1], hm_derivations[2],
        #                                 (FadeIn, [0,1]),
        #                                 (FadeIn, [4,5]),
        #                                 (FadeIn, [10,11]),
        #                                 ))
        
    
        # self.play(TransformWithBoxes(hm_derivations[2], hm_derivations[3],
        #                                 ([9,10,11,12], FadeOut, {"run_time": 0.5})))
        # self.play(TransformWithBoxes(hm_derivations[3], hm_derivations[4],
        #                                 (FadeIn, range(0,7)),
        #                                 (FadeIn, range(11,18), {"run_time": 0.45, "delay":0.5},),
        #                         ))
        # self.play(TransformWithBoxes(hm_derivations[4], hm_derivations[5],
        #                                 (range(11,21), FadeOut, {"run_time":0.5})))
    
        self.play(TransformWithBoxes(hm_derivations[5], hm_derivations[6],
                                        (FadeIn, [0]),
                                        (FadeIn, [12]),
                                        ))
        self.wait(2)
        return
        self.play(TransformWithBoxes(hm_derivations[6], hm_derivations[7],
                                        ([12,14], [13])))
