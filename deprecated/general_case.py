import numpy as np
from manim import *
from stitcher_scene import StitcherScene


class GeneralCase(StitcherScene):
    def construct_scene(self):

        with self.voiceover("So far we've looked at the case where n=3 and k=2 because we can visualize it. But in real life we will have more data than that, most likely.") as tracker:
            ... # TODO: Show X transforming from a 3x2 matrix to a bigger matrix 
        with self.voiceover("So let's talk about the general case, with arbitrarily many data points and predictors. Obviously, the animations are going to be wrong.") as tracker:
            ... # TODO: Show lots of vectors in the X space. They will not be linearly independent, but they would be if we had more dimensions.
        with self.voiceover("Y hat is equal to X beta hat. It's clear from this that Y hat could be anything in the column space of X.") as tracker:
            ... # TODO: Show the span of the vectors. I think I want to do this by, for every 2 vectors, showing a plane that contains those 2 vectors.
        with self.voiceover("And we choose, out of all the possible Y hats, the one that minimizes the sum of squared differences from Y, the Euclidean distance to Y.") as tracker:
            ... # TODO: Show searching many different points in the "span" and then settling on the one that is closest
        with self.voiceover("This is the orthogonal projection of Y onto the column space of X.") as tracker:
            ... # TODO: Show the perpendicular from Y to our "span" that's actually just a bunch of planes.
        with self.voiceover("So we want the matrix that's an orthogonal projection onto the column space of X.") as tracker:
            ...
