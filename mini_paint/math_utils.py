"""Vector math and 2D affine transformation helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Vec2 = np.ndarray


def vec2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float32)


def identity3() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


def translate2(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rotate2(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def scale2(sx: float, sy: float | None = None) -> np.ndarray:
    if sy is None:
        sy = sx
    return np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def apply2(matrix: np.ndarray, point: Vec2) -> Vec2:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    transformed = matrix @ homogeneous
    return transformed[:2]


def apply2_many(matrix: np.ndarray, points: Iterable[Vec2]) -> list[Vec2]:
    return [apply2(matrix, p) for p in points]


def distance(a: Vec2, b: Vec2) -> float:
    return float(np.linalg.norm(a - b))


def midpoint(a: Vec2, b: Vec2) -> Vec2:
    return (a + b) * 0.5


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def affine3_to_mat4(matrix: np.ndarray) -> np.ndarray:
    """Convert a 2D affine matrix to a 4x4 matrix."""
    return np.array(
        [
            [matrix[0, 0], matrix[0, 1], 0.0, matrix[0, 2]],
            [matrix[1, 0], matrix[1, 1], 0.0, matrix[1, 2]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def to_gl_mat3(matrix: np.ndarray) -> np.ndarray:
    """Pack a 3x3 affine matrix for glUniformMatrix3fv (column-major)."""
    return np.ascontiguousarray(matrix.T, dtype=np.float32).flatten()


def to_gl_mat4(matrix: np.ndarray) -> np.ndarray:
    """Pack a 4x4 matrix for glUniformMatrix4fv (column-major)."""
    return np.ascontiguousarray(matrix.T, dtype=np.float32).flatten()


def shape_centroid(vertices: list[Vec2]) -> Vec2:
    if not vertices:
        return vec2(0.0, 0.0)
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return vec2(sum(xs) / len(xs), sum(ys) / len(ys))
