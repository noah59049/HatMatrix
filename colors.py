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


# A command name, plus one immediately-following brace argument if present
# (e.g. matches "\end{bmatrix}" and "\vec{v}" as a whole, or bare "\Delta").
_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)(\{(?:[^{}]|\{[^{}]*\})*\})?")


def _protected_commands(tex_strings: list[str]) -> list[str]:
    """LaTeX commands (e.g. \\end, \\vec, \\Delta) whose name contains one of
    our bare-letter color targets as a substring. Isolating "e" globally
    would otherwise also match the "e" hiding inside \\end, \\vec, \\Delta,
    etc, splitting the command apart mid-name. Isolating the whole command
    (uncolored) first lets it win the match instead of the bare letter.

    Commands like \\begin/\\vec grab their next brace group as an argument,
    so the argument has to be isolated along with the command name - wrapping
    just "\\vec" and leaving "{v}" outside breaks \\vec's argument parsing.

    Scanned per tex_string (i.e. per positional arg to MathTex), not on all
    of them joined together - joining without the arg_separator can glue the
    tail of one arg onto the head of the next (e.g. "\\implies" + "H ..." ->
    "\\impliesH"), producing a bogus command that was never actually adjacent
    in the compiled tex.
    """
    letters = "".join(color_tex for color_tex, _ in _COLOR_TEX if len(color_tex) == 1)
    protected = set()
    for tex_string in tex_strings:
        for name, arg in _COMMAND_RE.findall(tex_string):
            if any(letter in name for letter in letters):
                protected.add("\\" + name + arg)
    return list(protected)


_BASE_ISOLATE = [tex for tex, _ in _COLOR_TEX]


class ColoredMathTex(MathTex):
    def __init__(self, *args, **kwargs):
        # Manim's own substring-isolation matcher uses "." without DOTALL, so
        # once any substring is isolated, a literal "\n" in the tex (e.g. from
        # numpy_to_latex's \begin{bmatrix}...\end{bmatrix} output) silently
        # truncates everything after it. The newlines are only there for
        # readability of the source string - LaTeX row breaks come from "\\",
        # not "\n" - so collapsing them to spaces is safe.
        args = tuple(
            re.sub(r"\s*\n\s*", " ", a) if isinstance(a, str) else a for a in args
        )
        isolate = _BASE_ISOLATE + _protected_commands([str(a) for a in args])
        kwargs.setdefault("substrings_to_isolate", isolate)
        super().__init__(*args, **kwargs)
        color_math(self)
