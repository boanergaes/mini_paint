"""Application state, tools, input handling, and main loop."""

from __future__ import annotations

import math
from enum import Enum, auto

import glfw
import numpy as np

from .hud import Hud, HudColumn
from .math_utils import apply2, clamp, distance, identity3, rotate2, scale2, shape_centroid, translate2, vec2
from .renderer import Renderer
from .shapes import LineShape, PolygonShape, PolylineShape, Shape
from .viewport import Viewport

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Mini Paint - Vector Drawing"


class Tool(Enum):
    LINE = auto()
    POLYLINE = auto()
    POLYGON = auto()
    SELECT = auto()
    TRANSFORM = auto()


class TransformMode(Enum):
    TRANSLATE = auto()
    ROTATE = auto()
    SCALE = auto()


class DrawColor(Enum):
    RED = (0.92, 0.28, 0.28)
    GREEN = (0.28, 0.82, 0.42)
    BLUE = (0.32, 0.58, 0.96)
    YELLOW = (0.95, 0.82, 0.22)
    CYAN = (0.28, 0.86, 0.9)
    MAGENTA = (0.86, 0.36, 0.86)
    WHITE = (0.95, 0.95, 0.95)


class MiniPaintApp:
    def __init__(self) -> None:
        self.viewport = Viewport()
        self.renderer: Renderer | None = None
        self.hud: Hud | None = None
        self.window = None

        self.shapes: list[Shape] = []
        self.selected_index: int | None = None
        self.tool = Tool.LINE
        self.transform_mode = TransformMode.TRANSLATE
        self.current_color = DrawColor.BLUE.value
        self.polygon_sides = 6

        self._draft_start: np.ndarray | None = None
        self._draft_points: list[np.ndarray] = []
        self._mouse_world = vec2(0.0, 0.0)
        self._dragging = False
        self._drag_origin = vec2(0.0, 0.0)
        self._initial_transform = identity3()
        self._transform_pivot = vec2(0.0, 0.0)
        self._rotate_anchor = 0.0

    def run(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.viewport.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.renderer = Renderer()
        self.hud = Hud(self.renderer)

        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_key_callback(self.window, self._on_key)

        while not glfw.window_should_close(self.window):
            self._render_frame()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.renderer.dispose()
        glfw.terminate()

    def _on_resize(self, _window, width: int, height: int) -> None:
        self.viewport.resize(width, height)

    def _on_cursor_move(self, _window, x_pos: float, y_pos: float) -> None:
        self._mouse_world = self.viewport.screen_to_world(x_pos, y_pos)
        if self._dragging and self.selected_index is not None:
            self._apply_live_transform()

    def _on_mouse_button(self, _window, button: int, action: int, mods: int) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
                self._finish_polyline()
            return

        if action == glfw.PRESS:
            self._handle_press(mods)
        elif action == glfw.RELEASE:
            self._handle_release()

    def _on_key(self, _window, key: int, _scancode: int, action: int, mods: int) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            self._clear_draft()
            return

        if key == glfw.KEY_1:
            self.tool = Tool.LINE
        elif key == glfw.KEY_2:
            self.tool = Tool.POLYLINE
        elif key == glfw.KEY_3:
            self.tool = Tool.POLYGON
        elif key == glfw.KEY_4:
            self.tool = Tool.SELECT
        elif key == glfw.KEY_5:
            self.tool = Tool.TRANSFORM
        elif key == glfw.KEY_T:
            self.transform_mode = TransformMode.TRANSLATE
        elif key == glfw.KEY_R and not (mods & glfw.MOD_CONTROL):
            self.transform_mode = TransformMode.ROTATE
        elif key == glfw.KEY_S and not (mods & glfw.MOD_CONTROL):
            self.transform_mode = TransformMode.SCALE
        elif key in (glfw.KEY_DELETE, glfw.KEY_BACKSPACE):
            self._delete_selected()
        elif key == glfw.KEY_LEFT_BRACKET:
            self.polygon_sides = max(3, self.polygon_sides - 1)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.polygon_sides = min(12, self.polygon_sides + 1)
        elif key == glfw.KEY_C:
            self._cycle_color()
        elif key == glfw.KEY_ENTER:
            self._finish_polyline()

    def _handle_press(self, mods: int) -> None:
        point = self._mouse_world.copy()

        if self.tool == Tool.LINE:
            if self._draft_start is None:
                self._draft_start = point
            else:
                self.shapes.append(
                    LineShape(color=self.current_color, start=self._draft_start.copy(), end=point.copy())
                )
                self._draft_start = None

        elif self.tool == Tool.POLYLINE:
            self._draft_points.append(point)

        elif self.tool == Tool.POLYGON:
            if self._draft_start is None:
                self._draft_start = point
            else:
                radius = distance(self._draft_start, point)
                rotation = math.atan2(point[1] - self._draft_start[1], point[0] - self._draft_start[0])
                self.shapes.append(
                    PolygonShape.create(
                        color=self.current_color,
                        center=self._draft_start.copy(),
                        radius=radius,
                        sides=self.polygon_sides,
                        rotation=rotation,
                    )
                )
                self._draft_start = None

        elif self.tool == Tool.SELECT:
            self._select_at(point)

        elif self.tool == Tool.TRANSFORM:
            if self.selected_index is None:
                self._select_at(point)
            if self.selected_index is not None:
                self._begin_transform(point)

    def _handle_release(self) -> None:
        if self.tool == Tool.TRANSFORM and self._dragging:
            self._dragging = False

    def _begin_transform(self, point: np.ndarray) -> None:
        shape = self.shapes[self.selected_index]
        self._dragging = True
        self._drag_origin = point.copy()
        self._initial_transform = shape.transform.copy()
        self._transform_pivot = _shape_pivot(self._initial_transform, shape)

        if self.transform_mode == TransformMode.ROTATE:
            self._rotate_anchor = math.atan2(
                point[1] - self._transform_pivot[1],
                point[0] - self._transform_pivot[0],
            )

    def _apply_live_transform(self) -> None:
        assert self.selected_index is not None
        shape = self.shapes[self.selected_index]
        current = self._mouse_world
        pivot = self._transform_pivot

        if self.transform_mode == TransformMode.TRANSLATE:
            delta = current - self._drag_origin
            shape.transform = translate2(delta[0], delta[1]) @ self._initial_transform

        elif self.transform_mode == TransformMode.ROTATE:
            current_angle = math.atan2(current[1] - pivot[1], current[0] - pivot[0])
            delta_angle = current_angle - self._rotate_anchor
            shape.transform = (
                translate2(pivot[0], pivot[1])
                @ rotate2(delta_angle)
                @ translate2(-pivot[0], -pivot[1])
                @ self._initial_transform
            )

        elif self.transform_mode == TransformMode.SCALE:
            start_offset = self._drag_origin - pivot
            current_offset = current - pivot
            start_distance = float(np.linalg.norm(start_offset))
            current_distance = float(np.linalg.norm(current_offset))
            uniform_scale = clamp(
                current_distance / max(start_distance, 1e-4),
                0.05,
                10.0,
            )

            if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
                sx = sy = uniform_scale
            else:
                sx = _positive_scale_ratio(current_offset[0], start_offset[0], uniform_scale)
                sy = _positive_scale_ratio(current_offset[1], start_offset[1], uniform_scale)

            shape.transform = (
                translate2(pivot[0], pivot[1])
                @ scale2(sx, sy)
                @ translate2(-pivot[0], -pivot[1])
                @ self._initial_transform
            )

    def _select_at(self, point: np.ndarray) -> None:
        tolerance = max(self.viewport.world_width, self.viewport.world_height) * 0.015
        for index in range(len(self.shapes) - 1, -1, -1):
            if self.shapes[index].hit_test_world(point, tolerance):
                self.selected_index = index
                return
        self.selected_index = None

    def _delete_selected(self) -> None:
        if self.selected_index is None:
            return
        del self.shapes[self.selected_index]
        self.selected_index = None

    def _finish_polyline(self) -> None:
        if self.tool != Tool.POLYLINE or len(self._draft_points) < 2:
            self._draft_points.clear()
            return
        self.shapes.append(
            PolylineShape(
                color=self.current_color,
                points=[point.copy() for point in self._draft_points],
            )
        )
        self._draft_points.clear()

    def _clear_draft(self) -> None:
        self._draft_start = None
        self._draft_points.clear()

    def _cycle_color(self) -> None:
        colors = [member.value for member in DrawColor]
        try:
            index = colors.index(self.current_color)
        except ValueError:
            index = 0
        self.current_color = colors[(index + 1) % len(colors)]

    def _render_frame(self) -> None:
        assert self.renderer is not None and self.hud is not None
        self.renderer.begin_frame(self.viewport, background=(0.1, 0.11, 0.14))
        self.renderer.draw_axes(self.viewport)

        for index, shape in enumerate(self.shapes):
            self.renderer.draw_shape(shape, selected=index == self.selected_index)

        if self.selected_index is not None:
            self.renderer.draw_selection_box(self.shapes[self.selected_index])

        self._draw_previews()
        self.hud.draw(self.viewport, self._hud_columns())

    def _draw_previews(self) -> None:
        if self.tool == Tool.LINE and self._draft_start is not None:
            self.renderer.draw_preview_polyline(
                [self._draft_start, self._mouse_world],
                self.current_color,
            )

        if self.tool == Tool.POLYLINE and self._draft_points:
            points = self._draft_points + [self._mouse_world]
            self.renderer.draw_preview_polyline(points, self.current_color)

        if self.tool == Tool.POLYGON and self._draft_start is not None:
            radius = distance(self._draft_start, self._mouse_world)
            rotation = math.atan2(
                self._mouse_world[1] - self._draft_start[1],
                self._mouse_world[0] - self._draft_start[0],
            )
            self.renderer.draw_preview_polygon(
                self._draft_start,
                radius,
                self.polygon_sides,
                rotation,
                self.current_color,
            )

    def _hud_columns(self) -> list[HudColumn]:
        tool_name = self.tool.name.title()
        transform_name = self.transform_mode.name.title()
        color_name = next(
            (member.name for member in DrawColor if member.value == self.current_color),
            "Custom",
        )
        selected = "None" if self.selected_index is None else str(self.selected_index + 1)
        return [
            HudColumn(
                title="STATUS",
                lines=[
                    f"Tool: {tool_name}",
                    f"Color: {color_name}",
                    f"Transform: {transform_name}",
                    f"Selected: {selected}",
                    f"Sides: {self.polygon_sides}",
                ],
            ),
            HudColumn(
                title="TOOLS",
                lines=[
                    "1  Line",
                    "2  Polyline",
                    "3  Polygon",
                    "4  Select",
                    "5  Transform",
                    "C  Cycle color",
                ],
            ),
            HudColumn(
                title="ACTIONS",
                lines=[
                    "T  Translate",
                    "R  Rotate",
                    "S  Scale (Shift uniform)",
                    "Del  Delete",
                    "Esc  Cancel",
                    "Enter  Finish polyline",
                ],
            ),
        ]


def _shape_pivot(transform: np.ndarray, shape: Shape) -> np.ndarray:
    return shape_centroid([apply2(transform, vertex) for vertex in shape.local_vertices])


def _positive_scale_ratio(current: float, start: float, fallback: float) -> float:
    if abs(start) < 1e-6:
        return fallback
    ratio = abs(current / start)
    return clamp(ratio, 0.05, 10.0)
