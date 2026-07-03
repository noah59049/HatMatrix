import numpy as np
from manim import *
from manim_voiceover import *
from manim_voiceover.services.stitcher import _StitcherService as StitcherService
import dir_config


class XSpan(VoiceoverScene):
    def construct(self):
        self.set_speech_service(StitcherService(
            dir_config.path_to_podcast("X_span"),
            cache_dir=dir_config.get_cache_dir(),
            min_silence_len=2000,
            keep_silence=(0, 0),
        ))
