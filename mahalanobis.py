import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class Mahalanobis(StitcherScene):
    def construct_scene(self):
        texes = [
            MathTex(r"\Sigma = I \implies D = ||\vec{x} - \vec{\mu}||"),
            MathTex(r"D(\vec{x},Q) = D(A \vec{x}, A Q)"),
            MathTex(r"D(\vec{x},Q) = D(\vec{a} + \vec{x}, \vec{a} + Q)"),
        ]
        VGroup(texes).arrange(DOWN, aligned_edge = LEFT)

        with self.voiceover("You may have noticed that the point with high leverage is an outlier, and the point with low leverage is near the center of the data. This is no accident, and in fact there's a mathematical relationship between leverage and the distance from the center of the X values, or more specifically called Mahalanobis distance.") as tracker:
            ...
        with self.voiceover("So let's suppose we have a bunch of vectors in R^j (or a distribution actually, but you can think of a set of vectors as a distribution too).") as tracker:
            ... # TODO: Show a 2D scatterplot, then transition it into a 3D graph of some kind of probability density function, then go back to the scatterplot.
        with self.voiceover("The best way I know of to think of Mahalanobis distances is a formula for distance with two properties. First, we want them to be equal to Euclidean distances when the covariance matrix is the identity matrix.") as tracker:
            self.play(FadeIn(texes[0]))
        with self.voiceover("Second, we want Mahalanobis distances to be invariant under linear transformations, in other words, adding something or multiplying by a matrix.") as tracker:
            self.play(FadeIn(texes[1]))
            self.play(FadeIn(texes[2]))
        with self.voiceover("From these two properties it's possible to derive the exact formula for Mahalanobis distance. First note that the Mahalanobis distance must necessarily be equal to the Euclidean distance after a whitening transformation is applied. A whitening transformation is one that turns the covariance matrix into the identity matrix.") as tracker:
            ...
        with self.voiceover("") as tracker:
            ...
        with self.voiceover("") as tracker:
            ...
        with self.voiceover("") as tracker:
            ...
        with self.voiceover("") as tracker:
            ...
        with self.voiceover("") as tracker:
            ...
