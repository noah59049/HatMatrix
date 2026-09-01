import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, X, n, leverages
from N_Tools import *
from colors import *


class LeveragesExplorer(StitcherScene):
    def construct_scene(self):

        # --- ArrayValueTrackers ---
        X1_tracker = ArrayValueTracker(X1)
        X_tracker = ArrayValueTracker(X)
        X_tracker.add_updater(lambda m: m.set_value(np.column_stack([as_col(np.ones(n)), X1_tracker.get_value()]))) 
        leverages_tracker = ArrayValueTracker(leverages)
        leverages_tracker.add_updater(lambda m : m.set_value(np.diagonal(X_tracker.get_value() @ np.linalg.inv(X_tracker.get_value().T @ X_tracker.get_value()) @ X_tracker.get_value().T)))
        self.add(X1_tracker, X_tracker, leverages_tracker)

        # --- Axes ---
        # We have 2 axes
        # One is a 1D axis with just the X variables
        # The other is a 2D axes with X and leverages
        # Shared x scale so identical X values sit at the same horizontal position
        # on both the standalone x axis and the X axis of leverages_axes.
        X_RANGE = [-0.5, 5.5, 1]
        X_LENGTH = 10

        # X axis
        x_axis = Axis1D(x_range = X_RANGE, x_length = X_LENGTH).to_edge(DOWN)
        x_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot(x_axis.c2p(x), color = X_COLOR) for x in X1_tracker.get_value().flatten()
                ]
            )
        )
        
        # Leverages axes
        leverages_axes = Axes(
            x_range = X_RANGE,
            x_length = X_LENGTH,
            y_range = [-0.2, 1.2],
        ).next_to(x_axis, UP)
        # Same range + length already gives both axes the same unit spacing;
        # align one shared x value horizontally so every x value lines up.
        leverages_axes.shift(
            (x_axis.c2p(0)[0] - leverages_axes.c2p(0, 0)[0]) * RIGHT
        )
        leverages_plot = always_redraw(
            lambda: VGroup(
                *[
                    Dot(leverages_axes.c2p(x, leverage), color = H_COLOR) for x, leverage in zip(X1_tracker.get_value().flatten(), leverages_tracker.get_value().flatten())
                ]
            )
        )

        # --- Parabola ---
        # So the relationship between X and leverage is a parabola, no matter what our X's are, and we are graphing that parabola.
        # So that's where the name parabola comes from

        def get_parabola_bhat():
            parabola_X = np.column_stack([X_tracker.get_value(), X1_tracker.get_value() * X1_tracker.get_value()])
            parabola_Y = as_col(leverages_tracker.get_value())
            parabola_bhat = np.linalg.inv(parabola_X.T @ parabola_X) @ parabola_X.T @ parabola_Y
            parabola_bhat = parabola_bhat.flatten()
            return parabola_bhat
        
        def parabola_vertex():
            parabola_bhat = get_parabola_bhat()
            # In general for a parabola y(x)
            # y = ax^2 + bx + c
            # y'(x) = 2ax + b
            # y'(x) = 0 iff x = -b/(2a)
            # The vertex is at x = -b/(2a)
            # And also at y = c
            # For this specific parabola
            # parabola_bhat = np.array([c, b, a])
            c, b, a = parabola_bhat
            vertex_x = -b / (2 * a)
            vertex_y = a * vertex_x ** 2 + b * vertex_x + c
            return vertex_x, vertex_y

        
        def parabola_yhat(parabola_graph_x_point):
            parabola_bhat = get_parabola_bhat()
            return parabola_bhat[0] + parabola_bhat[1] * parabola_graph_x_point + parabola_bhat[2] * parabola_graph_x_point **2
        
        parabola_ink = always_redraw(
            lambda: leverages_axes.plot(
                parabola_yhat,
                x_range = [X1_tracker.get_value().min(), X1_tracker.get_value().max()]
            )
        )
        parabola_vertex_dot = always_redraw(lambda: Dot(leverages_axes.c2p(*parabola_vertex())))

        # --- The actual scene being played ---
        self.add(
            x_axis,
            x_plot,
            leverages_axes,
            leverages_plot
        )
        self.wait(1)

        self.play(X1_tracker.animate.set_value_at((-1, 0), 2.5))

        self.wait(1)
        self.play(X1_tracker.animate.set_value(as_col(np.arange(1, 4, 0.1)[:X1.shape[0]])))
        self.wait(1)

        self.add(parabola_ink)
        self.add(parabola_vertex_dot)
        
        # Shift right
        self.play(X1_tracker.animate.set_value(X1_tracker.get_value() + 1))

        # Contract
        self.play(X1_tracker.animate.set_value(X1_tracker.get_value() * 0.8))

        # Move the rightmost value
        self.play(X1_tracker.animate.increase_value_at((-1, 0), 2.5))

        self.wait(1)