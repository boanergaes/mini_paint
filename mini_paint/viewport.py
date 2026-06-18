"""Window-to-world coordinate mapping."""

from __future__ import annotations

import numpy as np

from .math_utils import Vec2, vec2


class Viewport:
    """Maps screen pixels to a 2D world coordinate system."""

    def __init__(
        self,
        world_left: float = -10.0,
        world_right: float = 10.0,
        world_bottom: float = -7.5,
        world_top: float = 7.5,
    ) -> None:
        self.world_left = world_left
        self.world_right = world_right
        self.world_bottom = world_bottom
        self.world_top = world_top
        self.window_width = 1
        self.window_height = 1

    def resize(self, width: int, height: int) -> None:
        self.window_width = max(1, width)
        self.window_height = max(1, height)

    @property
    def world_width(self) -> float:
        return self.world_right - self.world_left

    @property
    def world_height(self) -> float:
        return self.world_top - self.world_bottom

    def screen_to_world(self, screen_x: float, screen_y: float) -> Vec2:
        """Convert GLFW screen coordinates (origin top-left) to world coordinates."""
        nx = screen_x / self.window_width
        ny = 1.0 - (screen_y / self.window_height)
        world_x = self.world_left + nx * self.world_width
        world_y = self.world_bottom + ny * self.world_height
        return vec2(world_x, world_y)

    def world_to_ndc(self, point: Vec2) -> Vec2:
        """Convert world coordinates to normalized device coordinates."""
        x = 2.0 * (point[0] - self.world_left) / self.world_width - 1.0
        y = 2.0 * (point[1] - self.world_bottom) / self.world_height - 1.0
        return vec2(x, y)

    def projection_matrix(self) -> np.ndarray:
        """Orthographic projection matching the configured world bounds."""
        left = self.world_left
        right = self.world_right
        bottom = self.world_bottom
        top = self.world_top
        return np.array(
            [
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
