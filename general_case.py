import numpy as np
from manim import *
from manim_voiceover import *
from manim_voiceover.services.stitcher import _StitcherService as StitcherService
import dir_config # type: ignore


class GeneralCase(VoiceoverScene):
    def construct(self):
        self.set_speech_service(StitcherService(
            dir_config.path_to_podcast("general_case"),
            cache_dir=dir_config.get_cache_dir(),
            min_silence_len=2000,
            keep_silence=(0, 0),
        ))

        with self.voiceover("So far we've looked at the case where n=3 and k=2 because we can visualize it. But in real life we will have more data than that, most likely.") as tracker:
            ...
        with self.voiceover("So let's talk about the general case.") as tracker:
            ...
        with self.voiceover("Y hat is equal to X beta hat. It's clear from this that Y hat could be anything in the column space of X.") as tracker:
            ...
        with self.voiceover("And we choose, out of all the possible Y hats, the one that minimizes the sum of squared differences from Y, the Euclidean distance to Y.") as tracker:
            ...
        with self.voiceover("This is the orthogonal projection of Y onto the column space of X.") as tracker:
            ...
        with self.voiceover("So we want the matrix that's an orthogonal projection onto the column space of X.") as tracker:
            ...
        with self.voiceover("In fancy linear algebra jargon, its eigenspace of 1 should be the column space of X, and its eigenspace of 0 should be the orthogonal complement of the column space, called the left null space.") as tracker:
            ...
