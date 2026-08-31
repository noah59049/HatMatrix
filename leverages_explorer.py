import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, X, n, leverages
from N_Tools import *


class LeveragesExplorer(StitcherScene):
    def construct_scene(self):
        X1_tracker = ArrayValueTracker(X1)
        X_tracker = ArrayValueTracker(X)
        X_tracker.add_updater(lambda m: m.set_value(np.column_stack([as_col(np.ones(n)), X1_tracker.get_value()]))) 
        leverages_tracker = ArrayValueTracker(leverages)
        leverages_tracker.add_updater(lambda m : m.set_value(np.diagonal(X_tracker.get_value() @ np.linalg.inv(X_tracker.get_value().T @ X_tracker.get_value()) @ X_tracker.get_value().T)))

        leverages_axes = Axes()
        leverages_plot = always_redraw(
            VGroup(
                *[
                    Dot(leverages_axes.c2p(x, leverage)) for x, leverage in zip(X1_tracker.get_value().flatten(), leverages_tracker.get_value().flatten())
                ]
            )
        )
        self.add(leverages_axes)
        self.add(leverages_plot)
        self.wait(2)
