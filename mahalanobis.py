import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class Mahalanobis(StitcherScene):
    def construct_scene(self):
        texes_list = [
            MathTex(r"\Sigma(Q) = \mathrm{I} \implies D = ||\vec{x} - \vec{\mu}||"),
            MathTex(r"D(\vec{x},Q) = D(A \vec{x}, A Q)"),
            MathTex(r"D(\vec{x},Q) = D(\vec{a} + \vec{x}, \vec{a} + Q)"),
        ]
        texes = VGroup(texes_list).arrange(DOWN, aligned_edge = LEFT)
        dots = VGroup(
            Dot().next_to(texes[0], LEFT), 
            Dot().next_to(texes[1], LEFT),
        )

        with self.voiceover("You may have noticed that the point with high leverage is an outlier, and the point with low leverage is near the center of the data. This is no accident, and in fact there's a mathematical relationship between leverage and the distance from the center of the X values, or more specifically called Mahalanobis distance.") as tracker:
            ...
        with self.voiceover("So let's suppose we have a bunch of vectors in R^j (or a distribution actually, but you can think of a set of vectors as a distribution too).") as tracker:
            ... # TODO: Show a 2D scatterplot, then transition it into a 3D graph of some kind of probability density function, then go back to the scatterplot.
        with self.voiceover("The best way I know of to think of Mahalanobis distances is a formula for distance with two properties. First, we want them to be equal to Euclidean distances when the covariance matrix is the identity matrix.") as tracker:
            self.play(
                FadeIn(texes[0]),
                FadeIn(dots[0])
            )
        with self.voiceover("Second, we want Mahalanobis distances to be invariant under linear transformations, in other words, adding something or multiplying by a matrix.") as tracker:
            self.play(
                FadeIn(texes[1]),
                FadeIn(dots[1])
            )
            self.play(FadeIn(texes[2]))

        whitening_texes = VGroup(
            MathTex(r"cov(W \vec{x}) = \mathrm{I}"),
            MathTex(r"E[W \vec{x} (W \vec{x})^T] = \mathrm{I}"),
            MathTex(r"E[W \vec{x} \vec{x}^T W^T] = \mathrm{I}"),
            MathTex(r"W E[\vec{x} \vec{x}^T] W^T = \mathrm{I}"),
            MathTex(r"W cov(\vec{x}) W^T = \mathrm{I}"),
            MathTex(r"cov(\vec{x}) W^T = W^{-1}"),
            MathTex(r"cov(\vec{x}) = W^{-1} {W^T}^{-1}"),
            MathTex(r"cov(\vec{x}) = (W^T W)^{-1}"),
        ).arrange(DOWN, aligned_edge = LEFT)

        with self.voiceover("From these two properties it's possible to derive the exact formula for Mahalanobis distance. First note that the Mahalanobis distance must necessarily be equal to the Euclidean distance after a whitening transformation is applied.") as tracker:
            self.play(
                FadeOut(texes),
                FadeOut(dots),
            )
        with self.voiceover("A whitening transformation is a linear transformation, so a matrix multiplication, that turns the covariance matrix into the identity matrix.") as tracker:
            self.play(FadeIn(whitening_texes[0]))
        with self.voiceover("Here I'll show a proof that the whitening matrix must yield the inverse of the covariance matrix when you multiply its transpose by it.") as tracker:
            self.play(FadeIn(whitening_texes))
        with self.voiceover("There are many different whitening matrices, but they will all yield the same Mahalanobis distances. One easy choice for a whitening matrix is one with the same eigenvectors as the covariance matrix, but with the inverse square root of the eigenvalues.") as tracker:
            self.wait(self.get_current_voiceover_duration() - 1.1)
            self.play(FadeOut(whitening_texes))
        
        formula_texes = VGroup(
            MathTex(r"D(\vec{x},Q) = D(W \vec{x},W Q)"),
            MathTex(r"D(\vec{x},Q) = ||W \vec{x} - W \mu||"),
            MathTex(r"D(\vec{x},Q) = ||W (\vec{x} - \mu)||"),
            MathTex(r"D(\vec{x},Q) = \sqrt{(W (\vec{x} - \mu))^T W (\vec{x} - \mu)}"),
            MathTex(r"D(\vec{x},Q) = \sqrt{(\vec{x} - \mu)^T W^T W (\vec{x} - \mu)}"),
            MathTex(r"D(\vec{x},Q) = \sqrt{(\vec{x} - \mu)^T cov(\vec{x})^{-1} (\vec{x} - \mu)}"),
        ).arrange(DOWN, aligned_edge = LEFT)
        with self.voiceover("Now we can use the fact that Mahalanobis distance is invariant under any transformation, which includes a whitening transformation.") as tracker:
            self.play(FadeIn(formula_texes[0]))
        with self.voiceover("But WQ has covariance matrix equal to the identity matrix, so Mahalanobis distances there are Euclidean distances from the mean, which is now W mu.") as tracker:
            self.play(FadeIn(formula_texes[1]))
            self.play(FadeIn(formula_texes[2]))
        with self.voiceover("Now we just expand the formula,") as tracker:
            self.play(FadeIn(formula_texes[3], run_time = self.get_current_voiceover_duration()))
        with self.voiceover("transpose of a product reverses their order,") as tracker:
            self.play(FadeIn(formula_texes[4], run_time = self.get_current_voiceover_duration()))
        with self.voiceover("and then use our result about W transpose W, and this is the formula for Mahalanobis distance.") as tracker:
            self.play(FadeIn(formula_texes[5], run_time = self.get_current_voiceover_duration()))
