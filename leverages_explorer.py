import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, leverages
from N_Tools import *


class LeveragesExplorer(StitcherScene):
    def construct_scene(self):
        leverages_axes = Axes()
        leverages_plot = VGroup(
            *[
                Dot(leverages_axes.c2p(x, leverage)) for x, leverage in zip(X1.flatten(), leverages.flatten())
            ]
        )
        self.add(leverages_axes)
        self.add(leverages_plot)
        self.wait(2)
