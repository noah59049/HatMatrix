import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from leverages import X1, X, n, leverages, rng
from N_Tools import *
from colors import *


class LeveragesExplorer(StitcherScene):
    def construct_scene(self):

        ############################################################
        ### Part 1: The value trackers and always_redraw objects ###
        ############################################################
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
        
        def parabola_yhat(parabola_graph_x_point):
            parabola_bhat = get_parabola_bhat()
            return parabola_bhat[0] + parabola_bhat[1] * parabola_graph_x_point + parabola_bhat[2] * parabola_graph_x_point **2
        
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
            return np.array([vertex_x, vertex_y])

        parabola_ink = always_redraw(
            lambda: leverages_axes.plot(
                parabola_yhat,
                x_range = [X1_tracker.get_value().min(), X1_tracker.get_value().max()]
            )
        )
        vertex_dot = always_redraw(lambda: Dot(leverages_axes.c2p(*parabola_vertex())))

        def get_vertex_text():
            return f"({parabola_vertex()[0]:.3f}, {parabola_vertex()[1]:.3f})"
        vertex_label = always_redraw(lambda: MathTex(get_vertex_text()).to_corner(UR))

        mean_line = always_redraw(
            lambda: DashedLine(
                leverages_axes.c2p(X1_tracker.get_value().mean(), 0),
                leverages_axes.c2p(X1_tracker.get_value().mean(), 1/3),
                color = GRAY
            ).set_opacity(0.5)
        )

        a_label = always_redraw(
            lambda: MathTex(
                f"a={get_parabola_bhat()[2]:.3f}"
            ).next_to(vertex_label, DOWN)
        )

        def get_variance(array):
            mu = np.mean(array)
            return np.mean(array * array) - mu * mu
        var_label = always_redraw(
            lambda: MathTex(
                f"\\sigma^2 = {get_variance(X1_tracker.get_value().flatten()):.3f}"
            ).next_to(a_label, DOWN)
        )

        var_times_a_label = always_redraw(
            lambda: MathTex(
                f"\\sigma^2 a = {get_variance(X1_tracker.get_value().flatten()) * get_parabola_bhat()[2]:.3f}"
            ).next_to(var_label, DOWN)
        )

        def get_horizontal_lines():
            lines = []
            for y_val in leverages_tracker.get_value().flatten():
                line = DashedLine(
                    leverages_axes.c2p(X_RANGE[0], y_val),
                    leverages_axes.c2p(X_RANGE[1], y_val),
                )
                line.set_opacity(0.3)
                lines.append(line)
            return VGroup(*lines)

        #######################################
        ### Part 2: Defining the animations ###
        #######################################
        def dot_dance(number, run_time = 1):
            if   number == 0: # Reset
                new_value = X1.copy()
            elif number == 1: # Decrease rightmost value
                new_value = X1_tracker.get_value()
                new_value[(-1, 0)] -= 2.5
            elif number == 2: # Become evenly spaced
                new_value = as_col(np.arange(1, 4, 0.1)[:X1.shape[0]])
            elif number == 3: # Shift right
                new_value = X1_tracker.get_value() + 1
            elif number == 4: # Expand
                X1_curr = X1_tracker.get_value()
                muX = X1_curr.mean()
                new_value = (X1_curr - muX) * 2 + muX
            elif number == 5: # Contract
                X1_curr = X1_tracker.get_value()
                muX = X1_curr.mean()
                new_value = (X1_curr - muX) * 0.5 + muX
            elif number < 10: # Random values
                if number in (6,9):
                    scale = 0.3
                elif number == 7:
                    scale = 0.84
                elif number == 8:
                    scale = 0.6
                new_value = rng.normal(
                    loc = number - 5, 
                    scale = scale, 
                    size = n
                )
                new_value = np.sort(new_value.flatten()).reshape(new_value.shape)
            else:
                raise ValueError("dot_dance got an invalid number")

            if run_time == 0:
                X1_tracker.set_value(new_value)
            else:
                self.play(X1_tracker.animate.set_value(new_value), run_time = run_time)

        # TODO: What are all the properties I want to explore and show?
        # The vertex is always the same y
        # Translating just translates the parabola
        # Stretching just stretches the parabola
        # The maximum you can reach is 1
        # If you keep the mean and standard deviation, you keep the parabola (I think).
        # If you plot the parabola in z-space, it is always the same.
        # The vertex of the parabola is always at x = mu.

        # That was a good list. Now what's the order?
        ### Start, move it in, evenly spaced
        # 1. It is a parabola
        ### Translate, stretch, move, random
        # 2. The vertex Y is always the same
        # 3. The vertex X is always at the mean
        # 4. The a of the parabola

        ##############################
        ### Part 3: Play the scene ###
        ##############################
        # --- The actual scene being played ---
        with self.voiceover("Now instead of the scatterplot of X and Y, we're going to just plot our X values; Y has no effect on leverages. And we're going to show") as tracker:
            self.add(
                x_axis,
                x_plot,
            )

        with self.voiceover("a graph with X values on the X-axis and leverages on the Y-axis. This will show how leverages change with X.") as tracker:
            self.add(
                leverages_axes,
                leverages_plot
            )

        with self.voiceover("Looks like the outlier is a high-leverage point, and all the others are low leverage. I wonder what happens if that outlier were") as tracker:
            ...

        with self.voiceover("closer to the distribution, say, if it were 2.5. Hmm, it looks like the leverages are making a curved shape. It kind of looks like a parabola. I wonder if it is a parabola. Let's") as tracker:
            dot_dance(1)

        with self.voiceover("fit a parabola to this and find out. That is a perfect fit to a parabola. Wow. What if we change the Xs to be") as tracker:
            self.play(IntroduceRedraw(parabola_ink, Create))

        with self.voiceover("uniformly spaced? Still a parabola.") as tracker:
            dot_dance(2)
        
        with self.voiceover("I have a hypothesis that the relationship between X and leverages is always a parabola. I want to track how the parabola depends on the Xs.") as tracker:
            self.play(IntroduceRedraw(vertex_dot))
        with self.voiceover("So let's add a dot for the vertex for the parabola. And let's show the coordinates of it.") as tracker:
            self.play(IntroduceRedraw(vertex_label))
        with self.voiceover("Let's also add a tracker for a, the coefficient in front of x squared that controls how wide or narrow the parabola is.") as tracker:
            self.play(IntroduceRedraw(a_label))
        with self.voiceover("And of course, if I'm wrong about it always being a parabola, then that will show here too.") as tracker:
            ...
        with self.voiceover("I wonder what happens if we") as tracker:
            ...
        with self.voiceover("increase all the Xs by a fixed amount") as tracker:
            dot_dance(3)
        with self.voiceover("Seems like the leverages were unaffected. The parabola just moved with the leverages.") as tracker:
            ...
        with self.voiceover("Now what happens if we") as tracker:
            ...
        with self.voiceover("scale the leverages? Seems like again, the leverages corresponding to each point didn't change.") as tracker:
            dot_dance(4)
        with self.voiceover("And if we draw a horizontal line at each point, you can see that each point stays locked on those horizontal lines as we scale.") as tracker:
            hlines = get_horizontal_lines()
            self.play(FadeIn(hlines))
            dot_dance(5)
            dot_dance(4)
            dot_dance(5)
            self.play(FadeOut(hlines))
        with self.voiceover("Now let's set the points to a couple of randomly chosen values and see what happens.") as tracker:
            for i in 6,7,8,9:
                dot_dance(i)
        
        with self.voiceover("One thing I noticed was that the y-coordinate, or maybe rather the leverage coordinate, if we can call it that, of the vertex of the parabola never changed.") as tracker:
            ...
        with self.voiceover("So let's keep a closer eye on the y-coordinate of the vertex.") as tracker:
            num_chars_kinda = len(get_vertex_text().split(",")[-1])
            vertex_y_coord_tex = vertex_label[0][-num_chars_kinda+1:-1]
            vertex_tex_rect = SurroundingRectangle(vertex_y_coord_tex)
            self.play(Create(vertex_tex_rect))
        with self.voiceover("And let's see what happens if the X values are everything they were before.") as tracker:
            for i in range(10):
                dot_dance(i, run_time = 0.5)
        with self.voiceover("Wow! The y-coordinate of the vertex of the parabola never changes.") as tracker:
            self.play(FadeOut(vertex_tex_rect))
        
        with self.voiceover("I also notice that the x-coordinate of the vertex of the parabola seems to be at the mean of the data.") as tracker:
            ...
        
        with self.voiceover("To test this, let's make a vertical line that tracks the mean of the data, and see if it always lines up with the vertex of the parabola.") as tracker:
            self.play(IntroduceRedraw(mean_line))

        with self.voiceover("Now let's cycle X through the same set of values we did before") as tracker:
            for i in range(10):
                dot_dance(i, run_time = 0.5)

        with self.voiceover("Indeed, the vertex of the parabola is always at the mean of X.") as tracker:
            ...

        with self.voiceover("Now my last question about the parabola is what affects a, the x squared term.") as tracker:
            ...

        with self.voiceover("It seems to be connected with the variance.") as tracker:
            ...

        with self.voiceover("When we translated our Xs, a didn't change at all.") as tracker:
            dot_dance(2, run_time = 0)
            self.wait(0.7)
            dot_dance(3)
        
        with self.voiceover("And when we scaled our Xs, increasing the variance, a decreased.") as tracker:
            dot_dance(4, run_time = 0)
            self.wait(0.7)
            dot_dance(5)

        with self.voiceover("So let's add a tracker for the variance and see what happens when we run through all the X values we had before.") as tracker:
            self.play(IntroduceRedraw(var_label))

        for i in range(10):
            dot_dance(i, run_time = 0.5)


# I have a suspicion that it has to do with the variance of the data.
# When we shifted all the Xs to the right, the x squared term of the parabola didn't change at all.
# When we shifted 
        


# TODO: I should probably explain at some point how I made this applet.
# It just requires applying the hat matrix to calculate leverages.