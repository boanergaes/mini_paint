"""Vector shape definitions stored as mathematical objects."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable

import numpy as np

from .math_utils import Vec2, apply2, distance, identity3, vec2


class ShapeKind(Enum):
    LINE = auto()
    POLYLINE = auto()
    POLYGON = auto()


@dataclass(kw_only=True)
class Shape(ABC):
    color: tuple[float, float, float]
    transform: np.ndarray = field(default_factory=identity3)

    @property
    @abstractmethod
    def kind(self) -> ShapeKind:
        raise NotImplementedError

    @property
    @abstractmethod
    def local_vertices(self) -> list[Vec2]:
        raise NotImplementedError

    def world_vertices(self) -> list[Vec2]:
        return [apply2(self.transform, vertex) for vertex in self.local_vertices]

    def model_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.transform[0, 0], self.transform[0, 1], 0.0, self.transform[0, 2]],
                [self.transform[1, 0], self.transform[1, 1], 0.0, self.transform[1, 2]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def hit_test_world(self, world_point: Vec2, tolerance: float) -> bool:
        inverse = np.linalg.inv(self.transform)
        local_point = apply2(inverse, world_point)
        return self.hit_test(local_point, tolerance)

    @abstractmethod
    def hit_test(self, point: Vec2, tolerance: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> Shape:
        raise NotImplementedError


@dataclass(kw_only=True)
class LineShape(Shape):
    start: Vec2
    end: Vec2

    @property
    def kind(self) -> ShapeKind:
        return ShapeKind.LINE

    @property
    def local_vertices(self) -> list[Vec2]:
        return [self.start, self.end]

    def hit_test(self, point: Vec2, tolerance: float) -> bool:
        return _distance_point_to_segment(point, self.start, self.end) <= tolerance

    def copy(self) -> LineShape:
        return LineShape(
            color=self.color,
            transform=self.transform.copy(),
            start=self.start.copy(),
            end=self.end.copy(),
        )


@dataclass(kw_only=True)
class PolylineShape(Shape):
    points: list[Vec2]

    @property
    def kind(self) -> ShapeKind:
        return ShapeKind.POLYLINE

    @property
    def local_vertices(self) -> list[Vec2]:
        return list(self.points)

    def hit_test(self, point: Vec2, tolerance: float) -> bool:
        if len(self.points) < 2:
            return False
        for index in range(len(self.points) - 1):
            if _distance_point_to_segment(point, self.points[index], self.points[index + 1]) <= tolerance:
                return True
        return False

    def copy(self) -> PolylineShape:
        return PolylineShape(
            color=self.color,
            transform=self.transform.copy(),
            points=[point.copy() for point in self.points],
        )


@dataclass(kw_only=True)
class PolygonShape(Shape):
    center: Vec2
    radius: float
    sides: int
    rotation: float = 0.0

    @property
    def kind(self) -> ShapeKind:
        return ShapeKind.POLYGON

    @property
    def local_vertices(self) -> list[Vec2]:
        return regular_polygon_vertices(self.center, self.radius, self.sides, self.rotation)

    def hit_test(self, point: Vec2, tolerance: float) -> bool:
        vertices = self.local_vertices
        if len(vertices) < 3:
            return False
        if _point_in_polygon(point, vertices):
            return True
        for index in range(len(vertices)):
            next_index = (index + 1) % len(vertices)
            if _distance_point_to_segment(point, vertices[index], vertices[next_index]) <= tolerance:
                return True
        return False

    def copy(self) -> PolygonShape:
        return PolygonShape(
            color=self.color,
            transform=self.transform.copy(),
            center=self.center.copy(),
            radius=self.radius,
            sides=self.sides,
            rotation=self.rotation,
        )


def regular_polygon_vertices(
    center: Vec2,
    radius: float,
    sides: int,
    rotation: float,
) -> list[Vec2]:
    vertices: list[Vec2] = []
    for index in range(sides):
        angle = rotation + (2.0 * math.pi * index / sides)
        vertices.append(
            vec2(
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
        )
    return vertices


def _distance_point_to_segment(point: Vec2, start: Vec2, end: Vec2) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        return distance(point, start)
    t = float(np.dot(point - start, segment) / length_sq)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return distance(point, projection)


def _point_in_polygon(point: Vec2, vertices: Iterable[Vec2]) -> bool:
    inside = False
    previous = None
    for current in vertices:
        if previous is None:
            previous = current
            continue
        intersects = (
            (current[1] > point[1]) != (previous[1] > point[1])
            and point[0]
            < (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1] + 1e-9)
            + current[0]
        )
        if intersects:
            inside = not inside
        previous = current
    return inside
