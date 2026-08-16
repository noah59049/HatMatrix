import numpy as np
from manim import *
from stitcher_scene import StitcherScene
from N_Tools import *


class Mahalanobis(StitcherScene, ThreeDScene):
    def construct_scene(self):
        self.renderer.camera.should_apply_shading = False

        cov = np.array([[1.4, 0.8], [0.8, 1.0]])
        mean = np.array([0.0, 0.0])
        inv_cov = np.linalg.inv(cov)
        norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

        def pdf(x, y):
            v = np.array([x, y]) - mean
            return norm_const * np.exp(-0.5 * v @ inv_cov @ v)

        rng = np.random.default_rng(3)
        L = np.linalg.cholesky(cov)
        samples = (L @ rng.standard_normal((2, 15))).T + mean

        PLOT_RADIUS = 4
        AXES_LENGTH = 5
        scatter_axes = Axes(
            x_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            y_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            x_length=AXES_LENGTH,
            y_length=AXES_LENGTH,
            tips=False,
        ).to_edge(LEFT, buff=0.17)
        scatter_dots = VGroup(*[
            Dot(scatter_axes.c2p(x, y), radius=0.05, color=BLUE_C)
            for x, y in samples
        ])
        scatter_group = VGroup(scatter_axes, scatter_dots)

        # A separate plain Axes (not ThreeDAxes) just to get an x/y base
        # mapping for the surface, positioned to the right of the
        # scatterplot -- ThreeDAxes crashes under FadeIn in this manim
        # version (see X_span.py's strip_sheen), so we sidestep it entirely
        # rather than work around it here.
        density_axes = Axes(
            x_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            y_range=[-PLOT_RADIUS, PLOT_RADIUS, 1],
            x_length=AXES_LENGTH,
            y_length=AXES_LENGTH,
            tips=False,
        ).to_edge(RIGHT, buff=0.75)
        HEIGHT_SCALE = 9.0
        density_surface = Surface(
            lambda u, v: density_axes.c2p(u, v) + OUT * pdf(u, v) * HEIGHT_SCALE,
            u_range=[-PLOT_RADIUS, PLOT_RADIUS],
            v_range=[-PLOT_RADIUS, PLOT_RADIUS],
            resolution=(32, 32),
            fill_opacity=0.85,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.5,
        )
        density_group = VGroup(density_axes, density_surface)

        with self.voiceover("Now let's suppose we have a distribution Q in R^k. This can either be a discrete distribution,") as tracker:
            # Locked to screen space so it stays flat and legible once the
            # camera tilts to show the 3D graph next to it below.
            self.add_fixed_in_frame_mobjects(scatter_group)
            self.play(FadeIn(scatter_group))

        rows = [[r"$\vec{x}$", r"$p(\vec{x})$"]]
        for vec in samples:
            coords = f"${numpy_to_latex(as_col(vec))}$"
            prob = r"$\frac{1}{n}$"
            row = [coords, prob]
            rows.append(row)
        table_text = latex_table(rows=rows)
        table_tex = Tex(table_text).scale_to_fit_height(7).next_to(scatter_axes, RIGHT)
        table_grid = extract_table_grid(table_tex)
        # self.play(FadeIn(table_tex))
        # print(f"{table_grid.keys()=}")
        # print(f"{len(rows)=}")
        # return
        def glyph_group(indices):
            # table_tex[0]'s submobjects are a plain Python list, which (unlike
            # a numpy array) only supports int/slice indexing -- not a list of
            # indices -- so pick the glyphs out one at a time instead.
            return VGroup(*[table_tex[0][i] for i in indices])

        remaining_indices = list(range(len(table_tex[0])))
        transforms = []
        for i, dot in enumerate(scatter_dots):
            glyph_indices = table_grid[i + 1, 0]
            transforms.append(TransformFromCopy(dot, glyph_group(glyph_indices)))
            for glyph_index in glyph_indices:
                remaining_indices.remove(glyph_index)

        with self.voiceover("which is basically sampling from a set of points,") as tracker:
            self.play(
                FadeIn(glyph_group(remaining_indices)),
                *transforms
            )
            self.wait(self.get_current_voiceover_duration() - 2.1)
            self.play(FadeOut(table_tex))

        with self.voiceover("or a continuous distribution with a PDF.") as tracker:
            half = self.get_current_voiceover_duration() / 2
            self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=0.6)
            self.play(FadeIn(density_group), run_time=half)
            self.play(FadeOut(density_group), run_time=half)
            self.move_camera(phi=0, theta=-90 * DEGREES)

        formula_tex = MathTex(
            r"D(\vec{x}, Q) = ", 
            r"\sqrt{(\vec{x} - \vec{\mu})^T cov(W)^{-1} (\vec{x} - \vec{\mu})}"
        ).to_corner(UL)

        mean_point = scatter_axes.c2p(*mean)

        def mahalanobis_distance(sample):
            v = sample - mean
            return float(np.sqrt(v @ inv_cov @ v))

        distances = np.array([mahalanobis_distance(s) for s in samples])
        highlighted_indices = range(len(scatter_dots))

        # Labels are kept as independent mobjects, never attached as a
        # dot's child. Mobject.get_center()/move_to() operate on a
        # mobject's whole family (itself + all submobjects), so a dot with
        # a label attached would have its "center" pulled toward the
        # label -- that's what was making the dashed lines the wrong
        # length/direction, and made move_to() below shift each dot by a
        # different, label-offset-dependent amount instead of moving the
        # cloud as one rigid body.
        distance_labels = VGroup()
        label_by_index = {}
        for i in highlighted_indices:
            dot = scatter_dots[i]
            label = Tex(f"{distances[i]:.2f}", font_size=24)
            label.next_to(dot, normalize(dot.get_center() - mean_point), buff=0.12)
            label.set_opacity(0)  # faded in below, once the dots are already on screen
            label_by_index[i] = label
            distance_labels.add(label)

        # The connecting lines, unlike the labels, do need to be recomputed
        # every frame: their far endpoint is whatever the dot's current
        # (possibly mid-animation) position is.
        distance_lines = always_redraw(lambda: VGroup(*[
            DashedLine(mean_point, scatter_dots[i].get_center(), color=GRAY, stroke_width=1.5)
            for i in highlighted_indices
        ]))

        def animate_transform(matrix, run_time=1.5, path_arc=0):
            """Moves every scatter dot to A @ sample. Labels aren't part of
            any dot's family (see above), so they're shifted by hand here,
            by the same delta as their dot -- a pure translation that
            leaves the printed number untouched.

            path_arc curves that motion along a circular arc instead of a
            straight line -- pass the actual rotation angle for a true
            rotation (path_along_arc derives the arc's center from the
            start/end points and this angle, and for points genuinely
            related by a rotation about mean_point, that derived center
            *is* mean_point). Leave it at 0 for reflections/scales, which
            really do move each point in a straight line.
            """
            transformed = samples @ matrix.T
            animations = []
            for i, (x, y) in enumerate(transformed):
                dot = scatter_dots[i]
                target = scatter_axes.c2p(x, y)
                if i in label_by_index:
                    animations.append(label_by_index[i].animate(path_arc=path_arc).shift(target - dot.get_center()))
                animations.append(dot.animate(path_arc=path_arc).move_to(target))
            self.play(*animations, run_time=run_time)

        with self.voiceover("The Mahalanobis distance is a measure of the standardized distance between a point and the mean of the distribution.") as tracker:
            i = 0
            for label in distance_labels:
                i += 1
                if i % 3 == 0:
                    label.set_opacity(1)
            self.play(FadeIn(distance_lines), FadeIn(distance_labels))
            self.play(Write(formula_tex[0]))
        with self.voiceover("It's given by this formula.") as tracker:
            self.play(Write(formula_tex[1]))

        properties = VGroup(
            MathTex(r"cov(Q) = I \implies D(\vec{x}, Q) = ||\vec{x} - \vec{\mu}||"),
            VGroup(
                MathTex(r"D(\vec{x}, Q) = D(A \vec{x}, A Q)"),
                Tex("if A is a rotation, reflection, or"),
                Tex("scales along the axes"),
            ).arrange(DOWN, aligned_edge = LEFT)
        ).arrange(DOWN, aligned_edge = LEFT)
        property_numbers = VGroup(*[
            Tex(f"{i}.").next_to(property, LEFT, aligned_edge=UP)
            for i, property in enumerate(properties, start=1)
        ])
        property_numbers.next_to(scatter_axes, RIGHT)
        properties.next_to(property_numbers, RIGHT, aligned_edge=UP)
        with self.voiceover("I think that the best way to explain that formula is to lead with two important properties that Mahalanobis distance should have, and then show how those properties necessitate this formula.") as tracker:
            self.play(FadeIn(property_numbers))
        with self.voiceover("The first property is that Mahalanobis distance is equal to Euclidean distance from the mean if the covariance matrix is equal to the identity matrix.") as tracker:
            self.play(Write(properties[0]))
        with self.voiceover("The second property is that rotating or reflecting the entire distribution should not change the Mahalanobis distance of any of the points, and neither should scaling by a nonzero amount along the coordinate axes. This property seems intuitive;") as tracker:
            self.play(Write(properties[1]))
        with self.voiceover("rotation and reflection should preserve measures of distance.") as tracker:
            theta = 50 * DEGREES
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            reflection = np.array([[-1.0, 0.0], [0.0, 1.0]])
            animate_transform(rotation, path_arc=theta)
            animate_transform(reflection @ rotation)
            animate_transform(np.eye(2))
        with self.voiceover("Scaling the distribution along the axes is also equivalent to changing the units, say measuring in meters instead of centimeters. It's intuitive that that shouldn't change any measures of distance.") as tracker:
            SCALE_FACTOR = 1.8
            scale = np.diag([SCALE_FACTOR, 1])
            # Stretching the x-axis's own tick spacing after the points
            # is what sells "this is a change of units" rather than "the
            # points just moved." about_point=mean_point is x=0, so the
            # y-axis (which sits at x=0) doesn't shift.
            animate_transform(scale)
            animate_transform(np.eye(2))
            self.play(scatter_axes.animate.stretch(SCALE_FACTOR, 0, about_point=mean_point))
            self.play(scatter_axes.animate.stretch(1 / SCALE_FACTOR, 0, about_point=mean_point))
        with self.voiceover("That's it. These 2 properties, which I think are pretty reasonable things for a distance measure to have, all you need to derive the formula for Mahalanobis distance, which we will do next. First, Mahalanobis distance doesn't change when you multiply the") as tracker: # TODO: Change the voiceover?
            ...
        with self.voiceover("distribution by any matrix. This is because") as tracker:
            cross_out_lines = VGroup(
                Line(properties[1][1].get_left() + LEFT * 0.1, properties[1][1].get_right() + RIGHT * 0.1),
                Line(properties[1][2].get_left() + LEFT * 0.1, properties[1][2].get_right() + RIGHT * 0.1),
            )
            self.play(Create(cross_out_lines))
            self.play(
                FadeOut(cross_out_lines),
                FadeOut(properties[1][1]),
                FadeOut(properties[1][2])
            )

        def highlight_region(image, x0, y0, x1, y1, color=YELLOW, opacity=0.4):
            """A translucent rectangle over the given region of `image`, like
            a highlighter over the screenshotted text. x0/x1 are horizontal
            fractions of image.width (0=left edge, 1=right edge); y0/y1 are
            vertical fractions of image.height (0=top edge, 1=bottom edge) --
            measured directly off the source PNG's pixel coordinates, so
            these stay correct regardless of how the image gets scaled or
            positioned on screen.
            """
            top_left = image.get_corner(UL)
            corner1 = top_left + RIGHT * x0 * image.width + DOWN * y0 * image.height
            corner2 = top_left + RIGHT * x1 * image.width + DOWN * y1 * image.height
            return Rectangle(
                width=abs(corner2[0] - corner1[0]),
                height=abs(corner2[1] - corner1[1]),
                fill_color=color,
                fill_opacity=opacity,
                stroke_width=0,
            ).move_to((corner1 + corner2) / 2)

        with self.voiceover("any invertible matrix can be written as a product of rotations and scaling the axes. I'm not going to prove this, but you can Google singular value decomposition if you're interested, and here we're just looking at the narrow case of real-valued square matrices.") as tracker:
            svd_image = ImageMobject("images/svd.png").scale_to_fit_width(11)
            self.add(svd_image)

            # "...a factorization of a real or complex matrix into a
            # rotation, followed by a scaling, followed by another
            # rotation." -- wraps across the article's first two lines, so
            # it takes two rectangles, one per line.
            svd_highlight = VGroup(
                highlight_region(svd_image, 0.1245, 0.032, 0.679, 0.088),
                highlight_region(svd_image, 0.0, 0.137, 0.505, 0.200),
            )
            self.play(
                Create(svd_highlight[0])
            )
            self.play(
                Create(svd_highlight[1])
            )