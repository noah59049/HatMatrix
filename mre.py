import numpy as np
from manim import *


class Bug(Scene):
    def construct(self):
        tracker = ValueTracker(np.nan)  # (1) starts at NaN

        circle = Circle(fill_opacity=1).set_sheen(0.3, UR)  # unrelated sibling mobject
        dot = always_redraw(lambda: Dot(np.array([tracker.get_value(), 0, 0])))
        group = VGroup(circle, dot)

        tracker.set_value(0.0)  # (2) ...but is fixed before anything is ever animated
        self.play(FadeIn(group))  # (3) crashes:
        # TypeError: LinearGradient.__new__() takes exactly 4 arguments (2 given)


class Control(Scene):
    """Identical, except tracker never touches NaN. Works fine."""
    def construct(self):
        tracker = ValueTracker(0.0)
        circle = Circle(fill_opacity=1).set_sheen(0.3, UR)
        dot = always_redraw(lambda: Dot(np.array([tracker.get_value(), 0, 0])))
        group = VGroup(circle, dot)
        self.play(FadeIn(group))  # fine
