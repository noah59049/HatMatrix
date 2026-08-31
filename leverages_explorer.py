import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, X, n, leverages
from N_Tools import *
from colors import *


class LeveragesExplorer(StitcherScene):
    def construct_scene(self):
        X1_tracker = ArrayValueTracker(X1)
        X_tracker = ArrayValueTracker(X)
        X_tracker.add_updater(lambda m: m.set_value(np.column_stack([as_col(np.ones(n)), X1_tracker.get_value()]))) 
        leverages_tracker = ArrayValueTracker(leverages)
        leverages_tracker.add_updater(lambda m : m.set_value(np.diagonal(X_tracker.get_value() @ np.linalg.inv(X_tracker.get_value().T @ X_tracker.get_value()) @ X_tracker.get_value().T)))
        self.add(X1_tracker, X_tracker, leverages_tracker)

        #TODO: Make the x axis and X axis of leverages have the same scale so the points line up
        x_axis = Axis1D().to_edge(DOWN)
        x_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot(x_axis.c2p(x), color = X_COLOR) for x in X1_tracker.get_value().flatten()
                ]
            )
        )
        leverages_axes = Axes(y_range = [-0.2, 1.2]).next_to(x_axis, UP)
        leverages_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot(leverages_axes.c2p(x, leverage), color = H_COLOR) for x, leverage in zip(X1_tracker.get_value().flatten(), leverages_tracker.get_value().flatten())
                ]
            )
        )
        self.add(
            x_axis,
            x_plot,
            leverages_axes,
            leverages_plot
        )
        self.wait(1)

        X1_ = X1.copy()
        X1_[-1,0] = 2.5
        self.play(X1_tracker.animate.set_value(X1_))

        self.wait(1)
        self.play(X1_tracker.animate.set_value(as_col(np.arange(1, 4, 0.1)[:X1.shape[0]])))
        self.wait(1)