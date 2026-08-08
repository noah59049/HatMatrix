import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from N_Tools import *


class Leverages(StitcherScene):
    def construct_scene(self):
        yhat_tex = MathTex(r"\hat{Y}=HY")
        deriv_tex = MathTex(r"\frac{\partial \hat{Y}_i} {\partial Y_i}=H_{ii}")
        limit_tex = MathTex(r"\lim_{\Delta Y_i \to 0} \frac{\Delta \hat{Y}_i} {\Delta Y_i} = H_{ii}")
        frac_tex = MathTex(r"\frac{\Delta \hat{Y}_i} {\Delta Y_i} = H_{ii}")

        X1 = [0.4, 0.7, 0.7, 1, 1, 1.2, 1.5, 1.59, 1.68, 1.73, 1.8, 5]
        n = len(X1)
        X1 = as_col(np.array(X1))
        X = np.column_stack([as_col(np.ones(n)), X1])
        bhat_ols = as_col(np.array([-1.842, 0.384]))
        yhat = X @ bhat_ols
        rng = np.random.default_rng(42)
        epsilon = as_col(rng.standard_normal(n))
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        e = epsilon - H @ epsilon
        Y = yhat + e
        bhat = np.linalg.inv(X.T @ X) @ X.T @ Y
        print(f"{bhat.shape=} {X.shape=} {yhat.shape=} {Y.shape=} {e.shape=} {epsilon.shape=} {bhat=}")        

        axes = Axes(
            x_range=[X1.min() - 0.5, X1.max() + 0.5, 1],
            y_range=[Y.min() - 0.5, Y.max() + 5, 1],
            x_length=4,
            y_length=4,
            tips=False,
        )

        points = VGroup([Dot(axes.c2p(x, y)) for x, y in zip(X1.flatten(), Y.flatten())])

        with self.voiceover("Now we're going to explain the relationship with leverages.") as tracker:
            ...
        with self.voiceover("Basically, Y hat equals H Y,") as tracker:
            self.play(Write(yhat_tex))
        with self.voiceover("so the derivative of Y hat i with respect to Y i is equal to Hii, the ith diagonal element of H.") as tracker:
            self.play(Write(deriv_tex))
        with self.voiceover("And in fact, we don't need any fancy 'differentiable functions are linear in the limit' arguments, it's just the change in your hat I over the change in Y i.") as tracker:
            self.play(TransformMatchingShapes(deriv_tex, limit_tex))
        with self.voiceover("Hii is therefore a measure of how much the ith datapoint can influence the regression coefficients by pulling the regression line or plane or surface towards itself.") as tracker:
            self.play(TransformMatchingShapes(limit_tex, frac_tex))
        with self.voiceover("Let's look at an example.") as tracker:
            self.play(
                FadeIn(axes), 
                FadeIn(points)
            )

        with self.voiceover("Here's a point with a high leverage value. Now let's look at what would happen if the Y for this point were to change. You can see that the regression line moves quite a bit, and Y hat gets pulled towards Y. ") as tracker:
            ...
        with self.voiceover("And in fact the amount that Y hat moves is exactly equal to the leverage value times the change in Y.") as tracker:
            ...
        with self.voiceover("Now let's look at this point, which has a much lower leverage value. Let's look at what would happen if the Y were to move.") as tracker:
            ...
        with self.voiceover("This time the regression line barely changes. The Y hat barely moves. Again, the change in Y hat is equal to the leverage times the change in Y, which this time is much lower.") as tracker:
            ...
        with self.voiceover("I've heard it said that the leverage is a measure of a point's potential to influence the regression coefficients. We don't say it's how influential it actually is because look at this point. It has a high leverage, but it's very close to the trend line, so it isn't actually moving it much, and if you were to remove it, the regression coefficients would stay basically the same.") as tracker:
            ...