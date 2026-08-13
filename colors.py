import re

from manim import *

Y_COLOR = ORANGE
YHAT_COLOR = YELLOW
X_COLOR = RED
BETA_COLOR = BLUE
H_COLOR = TEAL
E_COLOR = PURPLE

# (tex, color) pairs, most-specific first. "\hat{Y}" must be listed so that
# manim isolates it as one atomic unit before it ever considers isolating a
# bare "Y" - otherwise the Y inside \hat{Y} would get caught by the "Y" rule
# too and there'd be no way to tell the two apart.
_COLOR_TEX = [
    (r"\hat{Y}", YHAT_COLOR),
    (r"\hat{\beta}", BETA_COLOR),
    (r"\beta", BETA_COLOR),
    ("X", X_COLOR),
    ("Y", Y_COLOR),
    ("H", H_COLOR),
    ("e", E_COLOR),
]


def color_math(tex: MathTex) -> MathTex:
    for substring, color in _COLOR_TEX:
        tex.set_color_by_tex(substring, color, substring=False)
    return tex


def _protected_commands(full_tex: str) -> list[str]:
    """LaTeX commands (e.g. \\end, \\vec, \\Delta) that contain one of our
    bare-letter color targets as a substring. Isolating "e" globally would
    otherwise also match the "e" hiding inside \\end, \\vec, \\Delta, etc,
    splitting the command apart and coloring a fragment of it. Isolating the
    whole command name first (uncolored) lets it win the match instead."""
    letters = "".join(color_tex for color_tex, _ in _COLOR_TEX if len(color_tex) == 1)
    commands = set(re.findall(r"\\[a-zA-Z]+", full_tex))
    return [command for command in commands if any(letter in command[1:] for letter in letters)]


_BASE_ISOLATE = [tex for tex, _ in _COLOR_TEX]


class ColoredMathTex(MathTex):
    def __init__(self, *args, **kwargs):
        full_tex = "".join(str(a) for a in args)
        isolate = _BASE_ISOLATE + _protected_commands(full_tex)
        kwargs.setdefault("substrings_to_isolate", isolate)
        super().__init__(*args, **kwargs)
        color_math(self)
