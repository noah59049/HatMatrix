import numpy as np
from manim import *
from manim_voiceover import *
from manim_voiceover.services.stitcher import _StitcherService as StitcherService
import dir_config # type: ignore


class ArrayValueTracker(ValueTracker):
    """A ValueTracker that holds a numpy array of any shape instead of a single
    scalar, so it can be animated the same way with `.animate.set_value(new_array)`.

    Mobject (which ValueTracker subclasses) stores its state as `self.points`,
    an (N, 3) array of render points. We pack the tracked array's values into
    that same (N, 3) buffer (flattening/reshaping as needed) and remember the
    original shape so get_value() can hand back an array shaped like the input.
    """

    def __init__(self, array=None, **kwargs):
        Mobject.__init__(self, **kwargs)
        array = np.zeros(1) if array is None else np.array(array, dtype=float)
        self.array_shape = array.shape
        n_rows = max(1, int(np.ceil(array.size / 3)))
        self.points = np.zeros((n_rows, 3))
        self.set_value(array)

    def get_value(self):
        flat = self.points.flatten()[: np.prod(self.array_shape, dtype=int)]
        return flat.reshape(self.array_shape)

    def set_value(self, array):
        array = np.array(array, dtype=float)
        flat = self.points.flatten()
        flat[: array.size] = array.flatten()
        self.points = flat.reshape(self.points.shape)
        return self

    def increment_value(self, d_array):
        self.set_value(self.get_value() + np.array(d_array))
        return self


class XSpan(VoiceoverScene, ThreeDScene):
    def construct(self):
        self.set_speech_service(StitcherService(
            dir_config.path_to_podcast("X_span"),
            cache_dir=dir_config.get_cache_dir(),
            min_silence_len=2000,
            keep_silence=(0, 0),
        ))

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

        # n = 3, k = 2. Chosen arbitrarily: first column of X is all ones (intercept).
        n, k = 3, 2
        X = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0],
        ])
        Y = np.array([1.0, 3.0, 4.0])

        # Setup the 3D axes with the span of X
        bhat = ArrayValueTracker([0.0, 0.0])
        yhat = ArrayValueTracker(X @ bhat.get_value())
        yhat.add_updater(lambda m: m.set_value(X @ bhat.get_value()))
        self.add(bhat, yhat)

        axes = ThreeDAxes()
        point = always_redraw(lambda: Dot3D(axes.c2p(*yhat.get_value()), color=YELLOW, radius=0.1))
        y_point = Dot3D(axes.c2p(*Y), color=WHITE, radius=0.1)
        y_label = MathTex("Y").next_to(y_point, UP)

        def make_bhat_tex():
            return MathTex(
                r"\hat{\beta} = \begin{bmatrix}"
                + f"{bhat.get_value()[0]:.2f}"
                + r" \\ "
                + f"{bhat.get_value()[1]:.2f}"
                + r"\end{bmatrix}"
            ).to_edge(RIGHT)

        # always_redraw's mob.become(...) swaps in fresh submobjects each frame,
        # which drops out of add_fixed_in_frame_mobjects's tracked set, so we
        # re-register the fixed submobjects on every refresh instead.
        bhat_tex = make_bhat_tex()
        self.add_fixed_in_frame_mobjects(bhat_tex)

        def _refresh_bhat_tex(m):
            m.become(make_bhat_tex())
            self.add_fixed_in_frame_mobjects(m)

        bhat_tex.add_updater(_refresh_bhat_tex)

        with self.voiceover("X and Y are fixed at the time of data collection. But beta hat can vary.") as tracker:
            ...
        with self.voiceover("If beta hat is the zero vector, so is y hat.") as tracker:
            ...
        with self.voiceover("Now let's look at what happens if we vary beta zero hat.") as tracker:
            ...
        with self.voiceover("Y moves along this line in the direction of (1,1,1).") as tracker:
            ...
        with self.voiceover("Now let's look at what happens if we vary beta one hat.") as tracker:
            ...
        with self.voiceover("Y hat moves along in the direction of X1.") as tracker:
            ...
        with self.voiceover("So you can see how, by varying both beta zero hat and beta 1 hat, Y hat can be anything that's in the span of the two columns of matrix X.") as tracker:
            ...