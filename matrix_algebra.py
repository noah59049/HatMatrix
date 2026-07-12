import numpy as np
from manim import *
from manim_voiceover import *
from manim_voiceover.services.stitcher import _StitcherService as StitcherService
import dir_config # type: ignore


class MatrixAlgebra(VoiceoverScene):
    def construct(self):
        self.set_speech_service(StitcherService(
            dir_config.path_to_podcast("matrix_algebra"),
            cache_dir=dir_config.get_cache_dir(),
            min_silence_len=2000,
            keep_silence=(0, 0),
        ))

        with self.voiceover("Now it's time to try to derive a formula for the hat matrix.") as tracker:
            ...
        with self.voiceover("As it happens, I want to give a series of logical derivations that end with the hat matrix, but") as tracker:
            ...
        with self.voiceover("most of the 'derivations' end up being more like, here I just pulled the hat matrix out of the blue, now here's the proof as to why it's correct.") as tracker:
            ...
        with self.voiceover("So first thing, we want every column of X to be unchanged when multiplying by H.") as tracker:
            ...
        with self.voiceover("This is equivalent to saying that H X equals X.") as tracker:
            ...
        with self.voiceover("The matrix X is left alone if and only if every column of it is left alone.") as tracker:
            ...
        with self.voiceover("We also know that H times anything orthogonal to all the columns of X should be 0.") as tracker:
            ...
        with self.voiceover("Note that X transpose times anything orthogonal to all the columns of X is zero.") as tracker:
            ...
        with self.voiceover("So if H is equal to something times X transpose, it satisfies this second property that ") as tracker:
            ...
        