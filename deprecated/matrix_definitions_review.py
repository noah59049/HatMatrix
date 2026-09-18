import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class MatrixDefinitionsReview(StitcherScene):
    def construct_scene(self):
        
        with self.voiceover("Just to make sure that everyone is on the same page, I'm going to review our matrix definitions.") as tracker:
            ...
        with self.voiceover("Y is a column vector with our response variable.") as tracker:
            ...
        with self.voiceover("X is a matrix with our predictors. Each column represents one predictor, and the first column, the column of the ones, is for the intercept term.") as tracker:
            ...
        with self.voiceover("Each row of X represents one individual.") as tracker:
            ...
        with self.voiceover("Beta is a column vector with the true regression coefficients.") as tracker:
            ...
        with self.voiceover("Beta hat is our least squares estimate of the regression coefficients.") as tracker:
            ...
        with self.voiceover("So multiplying X by beta hat gives Y hat, our predictions of Y.") as tracker:
            ...

        with self.voiceover("The hat matrix is a matrix that if you multiply it by Y, you get Y hat.") as tracker:
            ...
        with self.voiceover("A funny way to say it is that it's a matrix that puts a hat on Y.") as tracker:
            ...
        
        with self.voiceover("When I first heard about the hat matrix, I was amazed that there is a matrix that you can just multiply by Y to get Y hat. It's so powerful it almost feels illegal.") as tracker:
            hm_def_tex = MathTex("HY=\hat{Y}")
            hm_def_tex.to_edge(UP)
            self.play(Write(hm_def_tex))
        with self.voiceover("But let's talk about this step by step.") as tracker:
            ...
