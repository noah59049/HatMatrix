from manim import *
from MF_Tools import *
from N_Tools import *

class YourScene(Scene):
    def construct(self):
        a = MathTex("a").to_corner(UL)
        b = MathTex("b")
        self.add(a)
        self.add(b)
        self.wait(1)
        group = VGroup(a,b)
        self.remove(group) # This doesn't remove anything
        self.wait(2)

        self.play(FadeOut(group)) # This does fade out the group
        self.wait(3)