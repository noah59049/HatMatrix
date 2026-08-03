from manim import *

class FlashBug(Scene):
    def construct(self):
        # A mobject that is only "supposed" to become visible partway through
        # the animation, via Create, and disappear again via FadeOut later.
        mystery_circle = Circle(color=RED)

        self.play(Succession(
            Wait(1),            # nothing should be on screen yet
            Create(mystery_circle),
            Wait(1),
            FadeOut(mystery_circle),
        ))
