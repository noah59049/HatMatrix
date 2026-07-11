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
        graph_group = VGroup(axes, point, y_point, y_label)
        
        # Rotate the graph group
        graph_group.rotate(50 * DEGREES, axis = UP)
        graph_group.rotate(62 * DEGREES, axis = LEFT)
        # TODO: Rotate the graph group so, instead of just guessing how to rotate it, it's rotated so the "camera angle" is orthogonal to both X0 and X1, for good viewing
        # Do not actually change the camera angle. Instead, just rotate the graph.

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
        # self.add_fixed_in_frame_mobjects(bhat_tex)

        def _refresh_bhat_tex(m):
            m.become(make_bhat_tex())
            self.add_fixed_in_frame_mobjects(m)

        bhat_tex.add_updater(_refresh_bhat_tex)

        with self.voiceover("X and Y are fixed at the time of data collection. But beta hat can vary.") as tracker:
            self.add(bhat_tex)
        with self.voiceover("If beta hat is the zero vector, so is y hat. Now let's look at what happens if") as tracker:
            self.add(graph_group)
        with self.voiceover("we vary beta zero hat. Y hat moves along this") as tracker:
            self.play(bhat.animate.set_value(np.array([ 1, 0])))
            self.play(bhat.animate.set_value(np.array([-1, 0])))
            self.play(bhat.animate.set_value(np.array([ 0, 0])))
        with self.voiceover("line in the direction of (1,1,1). Now let's look at what happens if") as tracker:
            X0_arr = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 0]), buff=0)
            self.play(FadeIn(X0_arr))
        with self.voiceover("we vary beta one hat. Y hat moves along") as tracker:
            self.play(bhat.animate.set_value(np.array([0, 1])))
            self.play(bhat.animate.set_value(np.array([0,-1])))
            self.play(bhat.animate.set_value(np.array([0, 0])))
        with self.voiceover("in the direction of X1.") as tracker:
            X1_arr = Arrow(axes.c2p(0, 0, 0), axes.c2p(*X[:, 1]), buff=0)
            self.play(FadeIn(X1_arr))
        with self.voiceover("So you can see how, by varying both beta zero hat and beta 1 hat, Y hat can be anything that's in the span of the two columns of X.") as tracker:
            ...