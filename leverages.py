import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from N_Tools import *
from colors import *


class Leverages(StitcherScene):
    def construct_scene(self):
        yhat_tex = MathTex(r"\hat{Y}=HY").to_edge(UP)
        deriv_tex = MathTex(r"\frac{\partial \hat{Y}_i} {\partial Y_i}=H_{ii}").next_to(yhat_tex, DOWN)
        limit_tex = MathTex(r"\lim_{\Delta Y_i \to 0} \frac{\Delta \hat{Y}_i} {\Delta Y_i} = H_{ii}").next_to(yhat_tex, DOWN)
        frac_tex = MathTex(r"\frac{\Delta \hat{Y}_i} {\Delta Y_i} = H_{ii}").next_to(yhat_tex, DOWN)

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

        Y_tracker = ArrayValueTracker(Y)
        bhat_tracker = ArrayValueTracker(bhat)
        bhat_tracker.add_updater(lambda m : m.set_value(np.linalg.inv(X.T @ X) @ X.T @ Y_tracker.get_value()))
        yhat_tracker = ArrayValueTracker(X @ bhat)
        yhat_tracker.add_updater(lambda m: m.set_value(X @ bhat_tracker.get_value()))
        self.add(Y_tracker, bhat_tracker, yhat_tracker)
        axes = Axes(
            x_range=[X1.min() - 0.5, X1.max() + 0.5, 1],
            y_range=[Y.min() - 0.5, Y.max() + 5, 1],
            x_length=8,
            y_length=6,
            tips=False,
        )
        points = always_redraw(lambda : VGroup([Dot(axes.c2p(x, y), color = Y_COLOR) for x, y in zip(X1.flatten(), Y_tracker.get_value().flatten())]))
        trendline = always_redraw(lambda: axes.plot(
            lambda x: bhat_tracker.get_value()[0,0] + bhat_tracker.get_value()[1,0] * x,
            color=YHAT_COLOR,
        ))

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
                FadeIn(points),
                FadeIn(trendline)
            )

        leverages = np.diagonal(H)
        i_big = np.argmax(leverages)
        i_small = np.argmin(leverages)
        with self.voiceover("Here's a point with a high leverage value. Now let's look at what would happen") as tracker:
            pt = points[i_big].copy()
            self.play(Indicate(pt))
            self.remove(pt)
        with self.voiceover("if the Y for this point were to change. You can see that the regression line moves quite a bit, and Y hat gets pulled towards Y. And in fact") as tracker:
            Y_bumped_big = Y.copy()
            Y_bumped_big[i_big,0] += 2
            self.play(Y_tracker.animate.set_value(Y_bumped_big))

        def make_stems(Y_bumped, i):
            Y_stem = Line(
                axes.c2p(X1[i,0],        Y[i,0]),
                axes.c2p(X1[i,0], Y_bumped[i,0]),
                color = Y_COLOR,
            )
            yhat_stem = DashedLine(
                axes.c2p(X1[i,0],           yhat[i,0]),
                axes.c2p(X1[i,0], (H @ Y_bumped)[i,0]),
                color = YHAT_COLOR,
            ).set_opacity(0.5)
            return Y_stem, yhat_stem
        class ExampleLeverageEquations(VGroup):
            def __init__(self, i, dy):
                self.tex1 = MathTex(r"\Delta \hat{Y} = ", "H_{ii}", r"\Delta Y").to_corner(UR)
                self.tex2 = MathTex(f"{(dy * leverages[i]):.2f} = ", f"{leverages[i]:.2f}", f" * {dy}").next_to(self.tex1, DOWN, aligned_edge=RIGHT)
                super().__init__(self.tex1, self.tex2)
            def draw_parts(self, i):
                return AnimationGroup(
                    Write(self.tex1[i]),
                    Write(self.tex2[i])
                )
        
        Y_stem_big, yhat_stem_big = make_stems(Y_bumped_big, i_big)
        equations_big = ExampleLeverageEquations(i_big, dy = 2)
        with self.voiceover("the amount that Y hat moves is exactly equal to") as tracker:
            self.play(
                FadeIn(yhat_stem_big),
                equations_big.draw_parts(0)
            )
        with self.voiceover("the leverage value times") as tracker:
            self.play(equations_big.draw_parts(1))
        with self.voiceover("the change in Y. Now let's look at") as tracker:
            self.play(
                equations_big.draw_parts(2),
                FadeIn(Y_stem_big)
            )
            self.play(
                FadeOut(Y_stem_big), 
                FadeOut(yhat_stem_big),
                FadeOut(equations_big),
                Y_tracker.animate.set_value(Y)
            )
        with self.voiceover("this point, which has a much lower leverage value. Let's look at what would happen") as tracker:
            pt = points[i_small].copy()
            self.play(Indicate(pt))
            self.remove(pt)
        with self.voiceover("if the Y were to move. This time the regression line barely changes.") as tracker:
            Y_bumped_small = Y.copy()
            Y_bumped_small[np.argmin(leverages),0] += 2
            self.play(Y_tracker.animate.set_value(Y_bumped_small))
        Y_stem_small, yhat_stem_small = make_stems(Y_bumped_small, i_small)
        equations_small = ExampleLeverageEquations(i_small, dy = 2)
        with self.voiceover("Y hat barely moves. Again,") as tracker:
            self.play(FadeIn(yhat_stem_small))
        with self.voiceover("the change in Y hat is equal to") as tracker:
            self.play(equations_small.draw_parts(0))
        with self.voiceover("the leverage times the") as tracker:
            self.play(equations_small.draw_parts(1))
        with self.voiceover("change in Y, which this time is much lower.") as tracker:
            self.play(
                FadeIn(Y_stem_small),
                equations_small.draw_parts(2)
            )
            self.play(
                FadeOut(Y_stem_small), 
                FadeOut(yhat_stem_small),
                FadeOut(equations_small),
                Y_tracker.animate.set_value(Y)
            )
        # with self.voiceover("I've heard it said that the leverage is a measure of a point's potential to influence the regression coefficients. We don't say it's how influential it actually is because look at this point. It has a high leverage, but it's very close to the trend line, so it isn't actually moving it much, and if you were to remove it, the regression coefficients would stay basically the same.") as tracker:
        #     ...