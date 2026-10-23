"""
Annotation geometry: shapes in the pixel coordinates of one image.

Coordinates are continuous pixel coordinates (x to the right, y down, pixel ``(col, row)`` covers
``[col, col + 1] x [row, row + 1]``), the convention of :mod:`scanbound.model.axes`.  Geometry is
pure data: it knows nothing about Qt, labels or what the shape means.  Physical lengths, areas and
angles take the pixel size ``(dx, dy)`` so anisotropic pixels are handled everywhere.

Shapes: :class:`Point`, :class:`Segment` (directed: start -> end), :class:`Polyline`,
:class:`Polygon`, :class:`Rect` (axis-aligned), :class:`Ellipse` (axis-aligned) and :class:`Mask`
(a painted group of pixels: the exact pixels, holes included, instead of an outline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Optional

import numpy as np

Scale = tuple[float, float]
_UNIT: Scale = (1.0, 1.0)


def _pts(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(arr)):
        raise ValueError("geometry coordinates must be finite")
    arr.flags.writeable = False
    return arr


def _scaled(points: np.ndarray, scale: Scale) -> np.ndarray:
    return points * np.asarray(scale, float)


def _path_length(points: np.ndarray, scale: Scale = _UNIT, closed: bool = False) -> float:
    p = _scaled(points, scale)
    if closed and len(p) > 1:
        p = np.vstack([p, p[:1]])
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def angle_deg(dx: float, dy: float, scale: Scale = _UNIT) -> float:
    """Direction of a pixel displacement in degrees, counter-clockwise from +x on screen, 0..360.
    Image y points down, so a displacement up the screen (negative dy) has a positive angle."""
    return math.degrees(math.atan2(-dy * scale[1], dx * scale[0])) % 360.0


class Geometry:
    """Base class.  Subclasses are frozen dataclasses holding an ``(N, 2)`` point array."""

    type: ClassVar[str] = ""
    closed: ClassVar[bool] = False
    min_points: ClassVar[int] = 1

    @property
    def points(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    # ---- shape measures (pixels unless a scale is given) -------------------------
    def bbox(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1)."""
        p = self.outline()
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def centroid(self) -> tuple[float, float]:
        c = self.points.mean(axis=0)
        return float(c[0]), float(c[1])

    def length(self, scale: Scale = _UNIT) -> float:
        """Path length (lines) or perimeter (closed shapes)."""
        return _path_length(self.outline(), scale, self.closed)

    def area(self, scale: Scale = _UNIT) -> float:
        return 0.0

    def outline(self) -> np.ndarray:
        """Points describing the drawn shape (polygonised for rects / ellipses)."""
        return self.points

    def is_degenerate(self) -> bool:
        if self.closed:
            return self.area() <= 1e-9
        if self.min_points >= 2:
            return self.length() <= 1e-9
        return False

    def contains(self, x: float, y: float) -> bool:
        if not self.closed:
            return False
        from matplotlib.path import Path
        return bool(Path(self.outline()).contains_point((x, y)))

    def distance_to(self, x: float, y: float) -> float:
        """Distance (px) from a point to the shape's outline (0 inside closed shapes)."""
        if self.closed and self.contains(x, y):
            return 0.0
        p = self.outline()
        if len(p) == 1:
            return float(math.hypot(p[0, 0] - x, p[0, 1] - y))
        segs = np.vstack([p, p[:1]]) if self.closed else p
        a, b = segs[:-1], segs[1:]
        ab = b - a
        t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        proj = a + ab * t[:, None]
        return float(np.hypot(proj[:, 0] - x, proj[:, 1] - y).min())

    def translated(self, dx: float, dy: float) -> "Geometry":
        return self.with_points(self.points + np.array([dx, dy]))

    def transformed(self, matrix: Any) -> "Geometry":
        """Apply a 3x3 affine matrix (pixel -> pixel) to every point."""
        m = np.asarray(matrix, float)
        p = np.column_stack([self.points, np.ones(len(self.points))]) @ m.T
        return self.with_points(p[:, :2])

    def with_points(self, points: Any) -> "Geometry":  # pragma: no cover - overridden
        raise NotImplementedError

    def mask(self, shape: tuple[int, int], line_width: float = 1.0) -> np.ndarray:
        """Bool mask of the pixels the shape covers (lines are drawn ``line_width`` px wide)."""
        import cv2
        out = np.zeros(shape, np.uint8)
        p = np.round((self.outline() - 0.5) * 16).astype(np.int32)   # pixel centres, 4 fractional bits
        if self.closed:
            cv2.fillPoly(out, [p], 1, lineType=cv2.LINE_8, shift=4)
        elif len(p) == 1:
            cv2.circle(out, tuple(int(v) for v in p[0]), int(max(1, line_width) * 8), 1, -1, shift=4)
        else:
            cv2.polylines(out, [p], False, 1, max(1, int(round(line_width))), lineType=cv2.LINE_8, shift=4)
        return out.astype(bool)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "points": self.points.round(6).tolist()}

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and np.array_equal(self.points, other.points)  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((self.type, self.points.round(6).tobytes()))


@dataclass(frozen=True, eq=False)
class Point(Geometry):
    x: float
    y: float
    type: ClassVar[str] = "point"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("point coordinates must be finite")

    @property
    def points(self) -> np.ndarray:
        return _pts([[self.x, self.y]])

    def with_points(self, points: Any) -> "Point":
        p = _pts(points)
        return Point(float(p[0, 0]), float(p[0, 1]))

    def bbox(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x, self.y


class _PointList(Geometry):
    _points: np.ndarray

    def __init__(self, points: Any):
        p = _pts(points)
        if len(p) < self.min_points:
            raise ValueError(f"a {self.type} needs at least {self.min_points} points, got {len(p)}")
        object.__setattr__(self, "_points", p)

    @property
    def points(self) -> np.ndarray:
        return self._points

    def with_points(self, points: Any) -> "_PointList":
        return type(self)(points)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._points.tolist()})"

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("geometry is immutable")


class Segment(_PointList):
    """A directed straight line from ``start`` to ``end`` (vectors, measurement lines)."""

    type = "segment"
    min_points = 2

    def __init__(self, points: Any):
        super().__init__(points)
        if len(self._points) != 2:
            raise ValueError("a segment has exactly 2 points")

    @classmethod
    def from_coords(cls, x0: float, y0: float, x1: float, y1: float) -> "Segment":
        return cls([[x0, y0], [x1, y1]])

    @property
    def start(self) -> tuple[float, float]:
        return float(self._points[0, 0]), float(self._points[0, 1])

    @property
    def end(self) -> tuple[float, float]:
        return float(self._points[1, 0]), float(self._points[1, 1])

    def delta(self) -> tuple[float, float]:
        (x0, y0), (x1, y1) = self.start, self.end
        return x1 - x0, y1 - y0

    def angle(self, scale: Scale = _UNIT) -> float:
        """Direction start -> end in degrees (0..360, counter-clockwise from +x on screen)."""
        return angle_deg(*self.delta(), scale)

    def reversed(self) -> "Segment":
        return Segment(self._points[::-1])


class Polyline(_PointList):
    """An open path (centre lines of ribbons, profile paths)."""

    type = "polyline"
    min_points = 2

    def end_to_end(self, scale: Scale = _UNIT) -> float:
        d = _scaled(self._points[-1:] - self._points[:1], scale)[0]
        return float(math.hypot(d[0], d[1]))

    def straightness(self) -> float:
        """End-to-end distance / path length (1 = straight)."""
        L = self.length()
        return 1.0 if L == 0 else min(1.0, self.end_to_end() / L)


class Polygon(_PointList):
    type = "polygon"
    closed = True
    min_points = 3

    def area(self, scale: Scale = _UNIT) -> float:
        return abs(_shoelace(self._points)) * scale[0] * scale[1]

    def centroid(self) -> tuple[float, float]:
        p = self._points
        a = _shoelace(p)
        if abs(a) < 1e-12:
            return super().centroid()
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        return float(((x + xn) * cross).sum() / (6 * a)), float(((y + yn) * cross).sum() / (6 * a))


@dataclass(frozen=True, eq=False)
class Rect(Geometry):
    """Axis-aligned rectangle; ``w`` / ``h`` are stored positive."""

    x: float
    y: float
    w: float
    h: float
    type: ClassVar[str] = "rect"
    closed: ClassVar[bool] = True

    def __post_init__(self) -> None:
        for name in ("x", "y", "w", "h"):
            object.__setattr__(self, name, float(getattr(self, name)))
        vals = (self.x, self.y, self.w, self.h)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("rect coordinates must be finite")
        if self.w < 0:
            object.__setattr__(self, "x", self.x + self.w)
            object.__setattr__(self, "w", -self.w)
        if self.h < 0:
            object.__setattr__(self, "y", self.y + self.h)
            object.__setattr__(self, "h", -self.h)

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Rect":
        return cls(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    @property
    def points(self) -> np.ndarray:
        return _pts([[self.x, self.y], [self.x + self.w, self.y + self.h]])

    def outline(self) -> np.ndarray:
        x0, y0, x1, y1 = self.x, self.y, self.x + self.w, self.y + self.h
        return _pts([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def bbox(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def area(self, scale: Scale = _UNIT) -> float:
        return self.w * self.h * scale[0] * scale[1]

    def centroid(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def with_points(self, points: Any) -> "Rect":
        p = _pts(points)
        if len(p) == 4:
            return Rect.from_corners(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        return Rect.from_corners(p[0, 0], p[0, 1], p[1, 0], p[1, 1])

    def transformed(self, matrix: Any) -> "Geometry":
        """Rotations turn a rect into a polygon; pure scale / shift keep it a rect."""
        m = np.asarray(matrix, float)
        if abs(m[0, 1]) < 1e-12 and abs(m[1, 0]) < 1e-12:
            return self.with_points(Geometry.transformed(Polygon(self.outline()), m).points)
        return Polygon(self.outline()).transformed(m)

    def iou(self, other: "Rect") -> float:
        ix = max(0.0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0.0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, eq=False)
class Ellipse(Rect):
    """Axis-aligned ellipse inscribed in the rectangle (x, y, w, h)."""

    type: ClassVar[str] = "ellipse"

    def outline(self, n: int = 48) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy = self.centroid()
        return _pts(np.column_stack([cx + self.w / 2 * np.cos(t), cy + self.h / 2 * np.sin(t)]))

    def area(self, scale: Scale = _UNIT) -> float:
        return math.pi * self.w * self.h / 4 * scale[0] * scale[1]

    def length(self, scale: Scale = _UNIT) -> float:
        a, b = self.w / 2 * scale[0], self.h / 2 * scale[1]
        h = ((a - b) / (a + b)) ** 2 if a + b else 0.0
        return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))     # Ramanujan

    def contains(self, x: float, y: float) -> bool:
        cx, cy = self.centroid()
        if self.w == 0 or self.h == 0:
            return False
        return ((x - cx) / (self.w / 2)) ** 2 + ((y - cy) / (self.h / 2)) ** 2 <= 1.0

    def with_points(self, points: Any) -> "Ellipse":
        r = Rect.with_points(self, points)
        return Ellipse(r.x, r.y, r.w, r.h)


class Mask(Geometry):
    """
    A set of whole pixels: ``bits`` (2D bool array) placed with its top-left pixel at the integer
    pixel ``(x0, y0)``.  Keeps the exact shape, holes and several pieces included, where an outline
    would round it off.  Measures are pixel counts: the area is the number of pixels, the perimeter
    the length of the pixel edges between the mask and the outside.  ``points`` are the corners of
    the bounding box (that is what moving drags); ``outline()`` is the outer boundary of the largest
    piece, along the pixel edges.
    """

    type: ClassVar[str] = "mask"
    closed: ClassVar[bool] = True
    min_points: ClassVar[int] = 2

    def __init__(self, x0: int, y0: int, bits: Any):
        b = np.asarray(bits, dtype=bool)
        if b.ndim != 2:
            raise ValueError("mask bits must be a 2D array")
        rows, cols = np.nonzero(b.any(axis=1))[0], np.nonzero(b.any(axis=0))[0]
        if len(rows):                                  # crop to the painted pixels
            b = b[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            x0, y0 = int(x0) + int(cols[0]), int(y0) + int(rows[0])
        b = np.ascontiguousarray(b)
        b.flags.writeable = False
        object.__setattr__(self, "x0", int(x0))
        object.__setattr__(self, "y0", int(y0))
        object.__setattr__(self, "bits", b)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("geometry is immutable")

    def __repr__(self) -> str:
        return f"Mask(x0={self.x0}, y0={self.y0}, {self.bits.shape[1]}x{self.bits.shape[0]}, {self.count} px)"

    @classmethod
    def from_image_mask(cls, mask: Any) -> "Mask":
        """A mask the size of the image (True = in the shape)."""
        return cls(0, 0, mask)

    @cached_property
    def count(self) -> int:
        return int(self.bits.sum())

    @property
    def points(self) -> np.ndarray:
        h, w = self.bits.shape
        p = np.array([[self.x0, self.y0], [self.x0 + w, self.y0 + h]], dtype=np.float64)
        p.flags.writeable = False
        return p

    def bbox(self) -> tuple[float, float, float, float]:
        h, w = self.bits.shape
        return float(self.x0), float(self.y0), float(self.x0 + w), float(self.y0 + h)

    def area(self, scale: Scale = _UNIT) -> float:
        return self.count * float(scale[0]) * float(scale[1])

    def length(self, scale: Scale = _UNIT) -> float:
        b = np.pad(self.bits, 1)
        vertical = int(np.count_nonzero(b[:, 1:] != b[:, :-1]))     # edges between left/right neighbours
        horizontal = int(np.count_nonzero(b[1:, :] != b[:-1, :]))
        return vertical * float(scale[1]) + horizontal * float(scale[0])

    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.bits)
        if not len(xs):
            return float(self.x0), float(self.y0)
        return float(xs.mean() + self.x0 + 0.5), float(ys.mean() + self.y0 + 0.5)

    def is_degenerate(self) -> bool:
        return self.count == 0

    def contains(self, x: float, y: float) -> bool:
        c, r = int(math.floor(x)) - self.x0, int(math.floor(y)) - self.y0
        h, w = self.bits.shape
        return 0 <= r < h and 0 <= c < w and bool(self.bits[r, c])

    @cached_property
    def contours(self) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        """(outer boundaries, hole boundaries), each an (N, 2) array along the pixel edges."""
        import cv2
        if not self.count:
            return (), ()
        big = np.kron(np.pad(self.bits, 1).astype(np.uint8), np.ones((2, 2), np.uint8))   # pixel corners exact
        found, hier = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer, holes = [], []
        for c, h in zip(found, hier[0]):
            p = c.reshape(-1, 2).astype(np.float64)
            p = np.floor((p + 1) / 2)                      # back to pixel-corner coordinates
            p += np.array([self.x0 - 1, self.y0 - 1])
            keep = np.ones(len(p), bool)                  # drop repeated corners
            keep[1:] = np.any(p[1:] != p[:-1], axis=1)
            p = p[keep]
            if len(p) > 1 and np.all(p[0] == p[-1]):
                p = p[:-1]
            if len(p) >= 3:
                (holes if h[3] >= 0 else outer).append(p)
        outer.sort(key=lambda q: -abs(_shoelace(q)))
        return tuple(outer), tuple(holes)

    def outline(self) -> np.ndarray:
        outer, _ = self.contours
        if outer:
            return outer[0]
        x0, y0, x1, y1 = self.bbox()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    def distance_to(self, x: float, y: float) -> float:
        if self.contains(x, y):
            return 0.0
        x0, y0, x1, y1 = self.bbox()
        gap = math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
        if gap > 0:
            return gap if gap > 3.0 else gap + 0.5         # outside the box: no closer than its edge
        r0, c0 = int(math.floor(y)) - self.y0, int(math.floor(x)) - self.x0
        ys, xs = np.nonzero(self.bits[max(r0 - 6, 0):r0 + 7, max(c0 - 6, 0):c0 + 7])
        if not len(xs):
            return 6.0
        xs = xs + max(c0 - 6, 0) + self.x0
        ys = ys + max(r0 - 6, 0) + self.y0
        dx = np.maximum(np.maximum(xs - x, x - xs - 1), 0)       # distance to each pixel square
        dy = np.maximum(np.maximum(ys - y, y - ys - 1), 0)
        return float(np.hypot(dx, dy).min())

    def translated(self, dx: float, dy: float) -> "Mask":
        return Mask(self.x0 + int(round(dx)), self.y0 + int(round(dy)), self.bits)

    def with_points(self, points: Any) -> "Mask":
        """Only a shift of the box moves a mask (its pixels are not resampled)."""
        p = np.asarray(points, float).reshape(-1, 2)
        return self.translated(p[0, 0] - self.x0, p[0, 1] - self.y0)

    def transformed(self, matrix: Any) -> "Mask":
        """The mask mapped by a 3x3 affine pixel -> pixel matrix (nearest pixel)."""
        import cv2
        m = np.asarray(matrix, float)
        if np.allclose(m[:2, :2], np.eye(2)):
            return self.translated(m[0, 2], m[1, 2])
        x0, y0, x1, y1 = self.bbox()
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]) @ m.T
        nx0, ny0 = np.floor(corners[:, :2].min(axis=0)).astype(int)
        nx1, ny1 = np.ceil(corners[:, :2].max(axis=0)).astype(int)
        # output pixel centre -> input pixel centre
        shift_out = np.array([[1, 0, nx0 + 0.5], [0, 1, ny0 + 0.5], [0, 0, 1.0]])
        shift_in = np.array([[1, 0, -self.x0 - 0.5], [0, 1, -self.y0 - 0.5], [0, 0, 1.0]])
        a = shift_in @ np.linalg.inv(m) @ shift_out
        out = cv2.warpAffine(self.bits.astype(np.uint8), a[:2], (max(int(nx1 - nx0), 1), max(int(ny1 - ny0), 1)),
                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        return Mask(int(nx0), int(ny0), out.astype(bool))

    def clipped(self, shape: tuple[int, int]) -> "Mask":
        """The part inside an image of ``shape`` (rows, cols)."""
        rows, cols = shape
        h, w = self.bits.shape
        c0, r0 = max(0, -self.x0), max(0, -self.y0)
        c1, r1 = min(w, cols - self.x0), min(h, rows - self.y0)
        if c1 <= c0 or r1 <= r0:
            return Mask(0, 0, np.zeros((0, 0), bool))
        return Mask(self.x0 + c0, self.y0 + r0, self.bits[r0:r1, c0:c1])

    def mask(self, shape: tuple[int, int], line_width: float = 1.0) -> np.ndarray:
        out = np.zeros(shape, bool)
        m = self.clipped(shape)
        if m.count:
            h, w = m.bits.shape
            out[m.y0:m.y0 + h, m.x0:m.x0 + w] = m.bits
        return out

    def combined(self, other: "Mask", subtract: bool = False) -> "Mask":
        """This mask with ``other``'s pixels added (or removed)."""
        x0, y0 = min(self.x0, other.x0), min(self.y0, other.y0)
        sx1, sy1, ox1, oy1 = self.bbox()[2], self.bbox()[3], other.bbox()[2], other.bbox()[3]
        x1, y1 = int(max(sx1, ox1)), int(max(sy1, oy1))
        if subtract:
            x0, y0, x1, y1 = self.x0, self.y0, int(sx1), int(sy1)
        out = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), bool)
        h, w = self.bits.shape
        out[self.y0 - y0:self.y0 - y0 + h, self.x0 - x0:self.x0 - x0 + w] = self.bits
        o = Mask(other.x0 - x0, other.y0 - y0, other.bits).clipped(out.shape)
        if o.count:
            oh, ow = o.bits.shape
            region = out[o.y0:o.y0 + oh, o.x0:o.x0 + ow]
            if subtract:
                region &= ~o.bits
            else:
                region |= o.bits
        return Mask(x0, y0, out)

    def to_dict(self) -> dict[str, Any]:
        h, w = self.bits.shape
        flat = self.bits.ravel()
        change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
        bounds = np.concatenate([[0], change, [flat.size]])
        runs = np.diff(bounds).tolist()
        if flat.size and flat[0]:
            runs = [0] + runs                          # runs alternate off / on, starting with off
        return {"type": self.type, "points": self.points.tolist(), "size": [w, h], "runs": runs}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mask":
        (x0, y0), = np.asarray(d["points"], float).reshape(-1, 2)[:1]
        w, h = (int(v) for v in d["size"])
        flat = np.zeros(w * h, bool)
        pos, on = 0, False
        for n in d.get("runs", []):
            n = int(n)
            if on:
                flat[pos:pos + n] = True
            pos += n
            on = not on
        return cls(int(round(x0)), int(round(y0)), flat.reshape(h, w))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mask) and (self.x0, self.y0) == (other.x0, other.y0) and \
            np.array_equal(self.bits, other.bits)

    def __hash__(self) -> int:
        return hash((self.type, self.x0, self.y0, self.bits.shape, np.packbits(self.bits).tobytes()))


GEOMETRY_TYPES: dict[str, type] = {"point": Point, "segment": Segment, "polyline": Polyline, "polygon": Polygon,
                                   "rect": Rect, "ellipse": Ellipse, "mask": Mask}


def geometry_from_dict(d: dict[str, Any]) -> Geometry:
    t = str(d.get("type"))
    pts = d.get("points")
    if t not in GEOMETRY_TYPES:
        raise ValueError(f"unknown geometry type {t!r}")
    if t == "mask":
        return Mask.from_dict(d)
    if t == "point":
        (x, y), = np.asarray(pts, float).reshape(-1, 2)[:1]
        return Point(float(x), float(y))
    if t in ("rect", "ellipse"):
        p = np.asarray(pts, float).reshape(-1, 2)
        cls = GEOMETRY_TYPES[t]
        return cls(float(p[0, 0]), float(p[0, 1]), float(p[1, 0] - p[0, 0]), float(p[1, 1] - p[0, 1]))
    return GEOMETRY_TYPES[t](pts)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clip_to_image(geom: Geometry, shape: tuple[int, int]) -> Optional[Geometry]:
    """The geometry with its points clamped into the image (None if nothing is left)."""
    rows, cols = shape
    if isinstance(geom, Mask):
        m = geom.clipped(shape)
        return None if m.is_degenerate() else m
    p = np.clip(geom.points, [0, 0], [cols, rows])
    try:
        g = geom.with_points(p)
    except ValueError:
        return None
    return None if g.is_degenerate() and geom.min_points > 1 else g
