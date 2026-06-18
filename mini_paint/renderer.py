"""OpenGL shader-based renderer for vector shapes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRIANGLE_FAN,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glClear,
    glClearColor,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glLineWidth,
    glShaderSource,
    glCompileShader,
    glAttachShader,
    glLinkProgram,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    glUniformMatrix4fv,
    glUniform3f,
)

from .math_utils import Vec2, vec2
from .shapes import Shape, ShapeKind
from .viewport import Viewport

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;

uniform mat4 uProjection;
uniform mat4 uModel;

void main() {
    vec4 world = uModel * vec4(aPos, 0.0, 1.0);
    vec4 clip = uProjection * world;
    gl_Position = vec4(clip.xy, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(uColor, 1.0);
}
"""


class Renderer:
    def __init__(self) -> None:
        self.program = _create_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.u_projection = glGetUniformLocation(self.program, "uProjection")
        self.u_model = glGetUniformLocation(self.program, "uModel")
        self.u_color = glGetUniformLocation(self.program, "uColor")
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 8, None)
        glBindVertexArray(0)

    def dispose(self) -> None:
        glDeleteBuffers(1, [self._vbo])
        glDeleteVertexArrays(1, [self._vao])
        glDeleteProgram(self.program)

    def begin_frame(self, viewport: Viewport, background: tuple[float, float, float]) -> None:
        glViewport(0, 0, viewport.window_width, viewport.window_height)
        glClearColor(*background, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.u_projection, 1, False, viewport.projection_matrix())
        glLineWidth(2.0)

    def draw_shape(self, shape: Shape, *, selected: bool = False) -> None:
        if not shape.local_vertices:
            return

        color = shape.color
        if selected:
            color = (
                min(1.0, color[0] + 0.25),
                min(1.0, color[1] + 0.25),
                min(1.0, color[2] + 0.25),
            )

        glUniformMatrix4fv(self.u_model, 1, False, shape.model_matrix())
        glUniform3f(self.u_color, *color)

        if shape.kind == ShapeKind.LINE:
            self._upload_vertices(shape.local_vertices)
            self._draw(GL_LINES, 2)
        elif shape.kind == ShapeKind.POLYLINE:
            self._upload_vertices(shape.local_vertices)
            self._draw(GL_LINE_STRIP, len(shape.local_vertices))
        elif shape.kind == ShapeKind.POLYGON:
            fill_vertices = shape.fill_vertices()
            outline_vertices = shape.outline_vertices()
            self._upload_vertices(fill_vertices)
            self._draw(GL_TRIANGLES, len(fill_vertices))
            glUniform3f(self.u_color, color[0] * 0.85, color[1] * 0.85, color[2] * 0.85)
            self._upload_vertices(outline_vertices)
            self._draw(GL_LINE_LOOP, len(outline_vertices))

    def draw_preview_polyline(self, points: list[Vec2], color: tuple[float, float, float]) -> None:
        if len(points) < 2:
            return
        identity = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(self.u_model, 1, False, identity)
        glUniform3f(self.u_color, *color)
        self._upload_vertices(points)
        self._draw(GL_LINE_STRIP, len(points))

    def draw_preview_polygon(
        self,
        center: Vec2,
        radius: float,
        sides: int,
        rotation: float,
        color: tuple[float, float, float],
    ) -> None:
        from .math_utils import affine3_to_mat4, translate2
        from .shapes import regular_polygon_vertices

        vertices = regular_polygon_vertices(vec2(0.0, 0.0), radius, sides, rotation)
        model = affine3_to_mat4(translate2(center[0], center[1]))
        glUniformMatrix4fv(self.u_model, 1, False, model)
        glUniform3f(self.u_color, *color)
        self._upload_vertices(vertices)
        self._draw(GL_LINE_LOOP, len(vertices))

    def draw_axes(self, viewport: Viewport) -> None:
        identity = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(self.u_model, 1, False, identity)
        axes = [
            vec2_pair(viewport.world_left, 0.0, viewport.world_right, 0.0),
            vec2_pair(0.0, viewport.world_bottom, 0.0, viewport.world_top),
        ]
        for axis, color in zip(axes, [(0.35, 0.35, 0.4), (0.35, 0.35, 0.4)]):
            glUniform3f(self.u_color, *color)
            self._upload_vertices(axis)
            self._draw(GL_LINES, 2)

    def draw_selection_box(self, shape: Shape) -> None:
        bounds = _bounds(shape.world_vertices())
        if bounds is None:
            return
        min_x, min_y, max_x, max_y = bounds
        padding = 0.25
        corners = [
            vec2(min_x - padding, min_y - padding),
            vec2(max_x + padding, min_y - padding),
            vec2(max_x + padding, max_y + padding),
            vec2(min_x - padding, max_y + padding),
        ]
        identity = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(self.u_model, 1, False, identity)
        glUniform3f(self.u_color, 1.0, 0.85, 0.2)
        self._upload_vertices(corners)
        self._draw(GL_LINE_LOOP, 4)

    def _draw(self, primitive: int, count: int) -> None:
        glBindVertexArray(self._vao)
        glDrawArrays(primitive, 0, count)

    def _upload_vertices(self, vertices: Iterable[Vec2]) -> None:
        data = np.array([[vertex[0], vertex[1]] for vertex in vertices], dtype=np.float32)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)


def vec2_pair(x1: float, y1: float, x2: float, y2: float) -> list[Vec2]:
    return [np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)]


def _bounds(vertices: list[Vec2]) -> tuple[float, float, float, float] | None:
    if not vertices:
        return None
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _create_program(vertex_source: str, fragment_source: str) -> int:
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(vertex_shader, vertex_source)
    glShaderSource(fragment_shader, fragment_source)
    glCompileShader(vertex_shader)
    glCompileShader(fragment_shader)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program
