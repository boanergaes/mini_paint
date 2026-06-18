"""Programmatic stroke font HUD rendered in screen space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL.GL import GL_LINE_LOOP, GL_LINES, GL_TRIANGLE_FAN, glUniform3f, glUniformMatrix4fv

from .math_utils import vec2
from .renderer import Renderer
from .viewport import Viewport

_FONT_HEIGHT = 8.0

_FONT = {
    " ": [],
    "-": [((0, 4), (6, 4))],
    "+": [((3, 1), (3, 7)), ((0, 4), (6, 4))],
    "/": [((0, 1), (6, 7))],
    ":": [((3, 2), (3, 3)), ((3, 5), (3, 6))],
    ".": [((3, 1), (3, 2))],
    "0": [((1, 1), (5, 1)), ((5, 1), (5, 7)), ((5, 7), (1, 7)), ((1, 7), (1, 1))],
    "1": [((3, 1), (3, 7)), ((2, 6), (3, 7)), ((3, 7), (4, 6))],
    "2": [((1, 7), (5, 7)), ((5, 7), (5, 4)), ((5, 4), (1, 1)), ((1, 1), (5, 1))],
    "3": [((1, 7), (5, 7)), ((5, 7), (5, 1)), ((1, 4), (4, 4)), ((1, 1), (5, 1))],
    "4": [((4, 1), (4, 7)), ((1, 5), (5, 5)), ((1, 7), (1, 5))],
    "5": [((5, 7), (1, 7)), ((1, 7), (1, 4)), ((1, 4), (5, 4)), ((5, 4), (5, 1)), ((5, 1), (1, 1))],
    "6": [((5, 7), (1, 7)), ((1, 7), (1, 1)), ((1, 1), (5, 1)), ((5, 1), (5, 4)), ((5, 4), (1, 4))],
    "7": [((1, 7), (5, 7)), ((5, 7), (2, 1))],
    "8": [((1, 1), (5, 1)), ((5, 1), (5, 7)), ((5, 7), (1, 7)), ((1, 7), (1, 1)), ((1, 4), (5, 4))],
    "9": [((1, 1), (5, 1)), ((5, 1), (5, 7)), ((5, 7), (1, 7)), ((1, 7), (1, 4)), ((1, 4), (5, 4))],
    "A": [((1, 1), (1, 7)), ((1, 7), (5, 7)), ((5, 7), (5, 1)), ((1, 4), (5, 4))],
    "B": [((1, 1), (1, 7)), ((1, 7), (4, 7)), ((4, 7), (5, 6)), ((5, 6), (4, 4)), ((4, 4), (1, 4)), ((4, 4), (5, 3)), ((5, 3), (4, 1)), ((4, 1), (1, 1))],
    "C": [((5, 7), (2, 7)), ((2, 7), (1, 6)), ((1, 6), (1, 2)), ((1, 2), (2, 1)), ((2, 1), (5, 1))],
    "D": [((1, 1), (1, 7)), ((1, 7), (4, 7)), ((4, 7), (5, 6)), ((5, 6), (5, 2)), ((5, 2), (4, 1)), ((4, 1), (1, 1))],
    "E": [((5, 7), (1, 7)), ((1, 7), (1, 1)), ((1, 1), (5, 1)), ((1, 4), (4, 4))],
    "F": [((1, 1), (1, 7)), ((1, 7), (5, 7)), ((1, 4), (4, 4))],
    "G": [((5, 7), (2, 7)), ((2, 7), (1, 6)), ((1, 6), (1, 2)), ((1, 2), (2, 1)), ((2, 1), (5, 1)), ((5, 1), (5, 4)), ((5, 4), (3, 4))],
    "H": [((1, 1), (1, 7)), ((5, 1), (5, 7)), ((1, 4), (5, 4))],
    "I": [((1, 7), (5, 7)), ((3, 7), (3, 1)), ((1, 1), (5, 1))],
    "K": [((1, 1), (1, 7)), ((5, 7), (1, 4)), ((2, 5), (5, 1))],
    "L": [((1, 7), (1, 1)), ((1, 1), (5, 1))],
    "M": [((1, 1), (1, 7)), ((1, 7), (3, 5)), ((3, 5), (5, 7)), ((5, 7), (5, 1))],
    "N": [((1, 1), (1, 7)), ((1, 7), (5, 1)), ((5, 1), (5, 7))],
    "O": [((1, 1), (5, 1)), ((5, 1), (5, 7)), ((5, 7), (1, 7)), ((1, 7), (1, 1))],
    "P": [((1, 1), (1, 7)), ((1, 7), (5, 7)), ((5, 7), (5, 4)), ((5, 4), (1, 4))],
    "Q": [((1, 1), (5, 1)), ((5, 1), (5, 6)), ((5, 6), (1, 7)), ((1, 7), (1, 1)), ((3, 3), (5, 1))],
    "R": [((1, 1), (1, 7)), ((1, 7), (5, 7)), ((5, 7), (5, 4)), ((5, 4), (1, 4)), ((2, 4), (5, 1))],
    "S": [((5, 7), (1, 7)), ((1, 7), (1, 4)), ((1, 4), (5, 4)), ((5, 4), (5, 1)), ((5, 1), (1, 1))],
    "T": [((1, 7), (5, 7)), ((3, 7), (3, 1))],
    "U": [((1, 7), (1, 2)), ((1, 2), (2, 1)), ((2, 1), (4, 1)), ((4, 1), (5, 2)), ((5, 2), (5, 7))],
    "V": [((1, 7), (3, 1)), ((3, 1), (5, 7))],
    "W": [((1, 7), (1, 1)), ((1, 1), (3, 4)), ((3, 4), (5, 1)), ((5, 1), (5, 7))],
    "X": [((1, 7), (5, 1)), ((5, 7), (1, 1))],
    "Y": [((1, 7), (3, 4)), ((5, 7), (3, 4)), ((3, 4), (3, 1))],
    "Z": [((1, 7), (5, 7)), ((5, 7), (1, 1)), ((1, 1), (5, 1))],
}


@dataclass
class HudColumn:
    title: str
    lines: list[str]


class Hud:
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer
        self.margin_px = 14
        self.column_width_px = 210
        self.line_height_px = 16
        self.char_width_px = 1.35
        self.char_height_px = 1.8

    def draw(self, viewport: Viewport, columns: list[HudColumn]) -> None:
        from .math_utils import to_gl_mat4

        identity4 = to_gl_mat4(np.eye(4, dtype=np.float32))
        glUniformMatrix4fv(self.renderer.u_projection, 1, False, identity4)
        self.renderer._set_model_identity()

        panel_height_px = self._panel_height(columns)
        panel_width_px = self.margin_px * 2 + len(columns) * self.column_width_px
        self._draw_panel_background(viewport, panel_width_px, panel_height_px)

        for column_index, column in enumerate(columns):
            x_px = self.margin_px + column_index * self.column_width_px
            if column_index > 0:
                separator_x = x_px - 8
                self._draw_separator(viewport, separator_x, panel_height_px)

            self._draw_text_px(
                viewport,
                x_px,
                self.margin_px,
                column.title,
                color=(0.72, 0.78, 0.9),
            )
            for line_index, line in enumerate(column.lines):
                y_px = self.margin_px + (line_index + 1) * self.line_height_px
                self._draw_text_px(
                    viewport,
                    x_px,
                    y_px,
                    line,
                    color=(0.9, 0.92, 0.96),
                )

    def _panel_height(self, columns: list[HudColumn]) -> float:
        max_lines = max(len(column.lines) for column in columns) if columns else 0
        return self.margin_px * 2 + (max_lines + 1) * self.line_height_px

    def _draw_panel_background(self, viewport: Viewport, panel_width_px: float, panel_height_px: float) -> None:
        width = viewport.window_width
        height = viewport.window_height
        left = self._px_to_ndc(0, 0, width, height)[0]
        top = self._px_to_ndc(0, 0, width, height)[1]
        right = self._px_to_ndc(panel_width_px, 0, width, height)[0]
        bottom = self._px_to_ndc(0, panel_height_px, width, height)[1]
        panel = [
            vec2(left, top),
            vec2(right, top),
            vec2(right, bottom),
            vec2(left, bottom),
        ]
        glUniform3f(self.renderer.u_color, 0.08, 0.09, 0.12)
        self.renderer._upload_vertices(panel)
        self.renderer._draw(GL_TRIANGLE_FAN, 4)
        glUniform3f(self.renderer.u_color, 0.22, 0.26, 0.34)
        self.renderer._upload_vertices(panel)
        self.renderer._draw(GL_LINE_LOOP, 4)

    def _draw_separator(self, viewport: Viewport, x_px: float, panel_height_px: float) -> None:
        width = viewport.window_width
        height = viewport.window_height
        top = self._px_to_ndc(x_px, self.margin_px // 2, width, height)
        bottom = self._px_to_ndc(x_px, panel_height_px - self.margin_px // 2, width, height)
        glUniform3f(self.renderer.u_color, 0.28, 0.32, 0.4)
        self.renderer._upload_vertices([top, bottom])
        self.renderer._draw(GL_LINES, 2)

    def _draw_text_px(
        self,
        viewport: Viewport,
        x_px: float,
        y_px: float,
        text: str,
        *,
        color: tuple[float, float, float],
    ) -> None:
        glUniform3f(self.renderer.u_color, *color)
        cursor_px = x_px
        for char in text.upper():
            segments = _FONT.get(char, [])
            for (x1, y1), (x2, y2) in segments:
                start = self._px_to_ndc(
                    cursor_px + x1 * self.char_width_px,
                    y_px + (_FONT_HEIGHT - y1) * self.char_height_px,
                    viewport.window_width,
                    viewport.window_height,
                )
                end = self._px_to_ndc(
                    cursor_px + x2 * self.char_width_px,
                    y_px + (_FONT_HEIGHT - y2) * self.char_height_px,
                    viewport.window_width,
                    viewport.window_height,
                )
                self.renderer._upload_vertices([start, end])
                self.renderer._draw(GL_LINES, 2)
            cursor_px += 7 * self.char_width_px

    @staticmethod
    def _px_to_ndc(x_px: float, y_px: float, width: int, height: int) -> np.ndarray:
        return vec2(
            2.0 * x_px / width - 1.0,
            1.0 - 2.0 * y_px / height,
        )
