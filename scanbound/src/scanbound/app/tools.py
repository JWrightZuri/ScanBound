"""
Drawing and editing tools: the interaction logic, separate from Qt.

The canvas converts mouse events to image pixel coordinates and calls ``press`` / ``move`` /
``release`` / ``double_click`` / ``key`` on the active tool.  A tool edits the annotation store
through the session (one undoable command per gesture) and exposes ``preview`` - geometries the
canvas draws while a gesture is in progress.  Tests drive tools with plain numbers.

Buttons are "left" / "right"; modifiers a set of "shift", "ctrl", "alt".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

import numpy as np

from ..annotations import Annotation, Ellipse, Geometry, Mask, Point, Polygon, Polyline, Rect, Segment, angle_deg

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session

Mods = frozenset


@dataclass
class Preview:
    geometry: Geometry
    style: str = "draft"          # "draft" (being drawn) / "edit" (being changed) / "band" (rubber band)
    color: Optional[str] = None


class Tool:
    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    shortcut: ClassVar[str] = ""
    hint: ClassVar[str] = ""
    cursor: ClassVar[str] = "cross"

    def __init__(self, session: "Session", tolerance: Callable[[], float] = lambda: 4.0):
        self.session = session
        self.tolerance = tolerance            # hit tolerance in image pixels (depends on the zoom)
        self.preview: list[Preview] = []

    # events (return True when handled)
    def press(self, x: float, y: float, button: str = "left", mods: frozenset = frozenset()) -> bool:
        return False

    def move(self, x: float, y: float, mods: frozenset = frozenset()) -> bool:
        return False

    def release(self, x: float, y: float, button: str = "left", mods: frozenset = frozenset()) -> bool:
        return False

    def double_click(self, x: float, y: float, mods: frozenset = frozenset()) -> bool:
        return False

    def key(self, key: str, mods: frozenset = frozenset()) -> bool:
        if key == "escape" and self.busy:
            self.cancel()
            return True
        return False

    def cancel(self) -> None:
        self.preview = []

    @property
    def busy(self) -> bool:
        return bool(self.preview)

    def deletable(self, a: Annotation) -> bool:
        """Whether Delete over ``a`` removes it while this tool is active (annotations this tool draws)."""
        return False

    def delete_at(self, x: float, y: float) -> bool:
        """Delete the annotation under (x, y) that this tool draws (the hovered one)."""
        if self.busy or not self.image_id:
            return False
        hits = [a for a in self.session.store.hit_test(self.image_id, x, y, self.tolerance()) if self.deletable(a)]
        if not hits:
            return False
        a = hits[0]
        self.session.store.remove([a.id], description=f"Delete {a.kind.replace('_', ' ')}")
        return True

    # helpers
    @property
    def image_id(self) -> str:
        return self.session.current_id

    def _shape(self) -> Optional[tuple[int, int]]:
        img = self.session.image()
        return img.shape if img is not None else None

    def _scale(self) -> tuple[float, float]:
        img = self.session.image()
        if img is None:
            return (1.0, 1.0)
        dx, dy = img.pixel_size
        return (1.0, dy / dx) if dx > 0 and dy > 0 else (1.0, 1.0)

    def _clip(self, x: float, y: float) -> tuple[float, float]:
        s = self._shape()
        if s is None:
            return x, y
        h, w = s
        return min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))

    def _add(self, kind: str, geom: Geometry, **kw) -> Optional[Annotation]:
        if not self.image_id:
            return None
        label = kw.pop("label", self.session.current_label)
        ann = Annotation(self.image_id, kind, geom, label=label, **kw)
        self.session.store.add(ann, f"Draw {kind.replace('_', ' ')}")
        self.session.select([ann.id])
        return ann


# =============================================================================
# Selection and editing
# =============================================================================

def handles(g: Geometry) -> np.ndarray:
    """Editable points of a geometry (corners for rects / ellipses; none for masks, which are painted)."""
    if isinstance(g, Mask):
        return np.zeros((0, 2), float)
    if isinstance(g, Rect):
        x0, y0, x1, y1 = g.bbox()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return np.asarray(g.points, float)


def with_handle(g: Geometry, k: int, x: float, y: float) -> Geometry:
    """``g`` with handle ``k`` moved to (x, y)."""
    if isinstance(g, Rect):
        h = handles(g)
        ox, oy = h[(k + 2) % 4]                        # the opposite corner stays
        return type(g).from_corners(ox, oy, x, y)
    p = np.array(g.points, float)
    p[k] = (x, y)
    return g.with_points(p)


class SelectTool(Tool):
    name = "select"
    title = "Select / edit"
    shortcut = "S"
    hint = ("Click to select (Shift adds), drag to move, drag a handle to reshape, drag on empty space to select "
            "an area. Double-click an outline to add a vertex, Alt+click a vertex to delete it. Del deletes.")
    cursor = "arrow"
    drag_threshold = 1.5

    def __init__(self, session: "Session", tolerance: Callable[[], float] = lambda: 4.0):
        super().__init__(session, tolerance)
        self._mode = ""
        self._start = (0.0, 0.0)
        self._target: Optional[Annotation] = None
        self._handle = -1
        self._moved = False

    def _handle_at(self, x: float, y: float) -> tuple[Optional[Annotation], int]:
        tol = self.tolerance()
        best = (None, -1, tol)
        for a in self.session.selected():
            if a.image_id != self.image_id:
                continue
            for k, (hx, hy) in enumerate(handles(a.geometry)):
                d = math.hypot(hx - x, hy - y)
                if d <= best[2]:
                    best = (a, k, d)
        return best[0], best[1]

    def press(self, x, y, button="left", mods=frozenset()):
        if button != "left" or not self.image_id:
            return False
        self._start, self._moved = (x, y), False
        a, k = self._handle_at(x, y)
        if a is not None:
            if "alt" in mods and not isinstance(a.geometry, (Rect, Point)):
                self._delete_vertex(a, k)
                self._mode = ""
                return True
            self._mode, self._target, self._handle = "handle", a, k
            return True
        hits = self.session.store.hit_test(self.image_id, x, y, self.tolerance())
        hits = [h for h in hits if self._visible(h)]
        if hits:
            hit = hits[0]
            if "shift" in mods or "ctrl" in mods:
                self.session.select([hit.id], toggle=True)
            elif hit.id not in self.session.selection:
                self.session.select([hit.id])
            self._mode, self._target = "move", hit
            return True
        self._mode = "band"
        if not ({"shift", "ctrl"} & mods):
            self.session.select([])
        return True

    def _visible(self, a: Annotation) -> bool:
        hidden = getattr(self.session, "hidden_statuses", ())
        return a.status not in hidden

    def move(self, x, y, mods=frozenset()):
        if not self._mode:
            return False
        sx, sy = self._start
        if not self._moved and math.hypot(x - sx, y - sy) < self.drag_threshold * self.tolerance() / 4:
            return True
        self._moved = True
        if self._mode == "handle" and self._target is not None:
            cur = self.session.store.get(self._target.id)
            if cur is not None:
                self.preview = [Preview(with_handle(cur.geometry, self._handle, *self._clip(x, y)), "edit")]
        elif self._mode == "move":
            dx, dy = x - sx, y - sy
            self.preview = [Preview(a.geometry.translated(dx, dy), "edit") for a in self.session.selected()]
        elif self._mode == "band":
            self.preview = [Preview(Rect.from_corners(sx, sy, x, y), "band")]
        return True

    def release(self, x, y, button="left", mods=frozenset()):
        if not self._mode:
            return False
        mode, self._mode = self._mode, ""
        self.preview = []
        sx, sy = self._start
        store = self.session.store
        if not self._moved:
            return True
        if mode == "handle" and self._target is not None:
            cur = store.get(self._target.id)
            if cur is not None:
                g = with_handle(cur.geometry, self._handle, *self._clip(x, y))
                if not g.is_degenerate():
                    store.update(cur.id, "Reshape annotation", geometry=g)
        elif mode == "move":
            ids = list(self.session.selection)
            if ids and (x, y) != (sx, sy):
                store.move(ids, x - sx, y - sy)
        elif mode == "band":
            band = Rect.from_corners(sx, sy, x, y)
            inside = [a.id for a in store.for_image(self.image_id) if self._visible(a) and _inside(a.geometry, band)]
            self.session.select(inside, add=bool({"shift", "ctrl"} & mods))
        return True

    def double_click(self, x, y, mods=frozenset()):
        hits = [h for h in self.session.store.hit_test(self.image_id, x, y, self.tolerance()) if self._visible(h)]
        for a in hits:
            g = a.geometry
            if isinstance(g, (Polygon, Polyline)):
                p = np.asarray(g.points, float)
                segs = np.vstack([p, p[:1]]) if g.closed else p
                best, k = math.inf, -1
                for i in range(len(segs) - 1):
                    d = _seg_dist(segs[i], segs[i + 1], x, y)
                    if d < best:
                        best, k = d, i
                if best <= self.tolerance():
                    new = np.insert(p, k + 1, [x, y], axis=0)
                    self.session.store.update(a.id, "Add vertex", geometry=g.with_points(new))
                    self.session.select([a.id])
                    return True
        return False

    def _delete_vertex(self, a: Annotation, k: int) -> None:
        g = a.geometry
        p = np.delete(np.asarray(g.points, float), k, axis=0)
        if len(p) >= g.min_points and len(p) >= (3 if g.closed else 2):
            self.session.store.update(a.id, "Delete vertex", geometry=g.with_points(p))

    def key(self, key, mods=frozenset()):
        store = self.session.store
        sel = list(self.session.selection)
        if key in ("delete", "backspace") and sel:
            store.remove(sel, description=f"Delete {len(sel)} annotation(s)")
            return True
        if key == "escape":
            if self.busy or self._mode:
                self._mode = ""
                self.cancel()
            else:
                self.session.select([])
            return True
        step = 10.0 if "shift" in mods else 1.0
        arrows = {"left": (-step, 0), "right": (step, 0), "up": (0, -step), "down": (0, step)}
        if key in arrows and sel:
            store.move(sel, *arrows[key])
            return True
        return False


def _inside(g: Geometry, band: Rect) -> bool:
    x0, y0, x1, y1 = g.bbox()
    return x0 >= band.x and y0 >= band.y and x1 <= band.x + band.w and y1 <= band.y + band.h


def _seg_dist(a: np.ndarray, b: np.ndarray, x: float, y: float) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1))
    px, py = a + t * ab
    return math.hypot(px - x, py - y)


# =============================================================================
# Drawing tools
# =============================================================================

class _DragTool(Tool):
    """Press - drag - release creates one shape."""
    min_size = 1.0

    def __init__(self, session, tolerance=lambda: 4.0):
        super().__init__(session, tolerance)
        self._start: Optional[tuple[float, float]] = None

    def press(self, x, y, button="left", mods=frozenset()):
        if button != "left" or not self.image_id:
            return False
        self._start = self._clip(x, y)
        return True

    def move(self, x, y, mods=frozenset()):
        if self._start is None:
            return False
        g = self.make(self._start, self._clip(x, y), mods)
        self.preview = [Preview(g, "draft", self.session.label_color(self.session.current_label))] if g else []
        return True

    def release(self, x, y, button="left", mods=frozenset()):
        if self._start is None:
            return False
        start, self._start = self._start, None
        self.preview = []
        g = self.make(start, self._clip(x, y), mods)
        if g is not None and self._big_enough(start, self._clip(x, y)):
            self.commit(g)
        return True

    def cancel(self):
        self._start = None
        super().cancel()

    def _big_enough(self, a, b) -> bool:
        return math.hypot(b[0] - a[0], b[1] - a[1]) >= self.min_size * max(1.0, self.tolerance() / 4)

    def make(self, a, b, mods) -> Optional[Geometry]:  # pragma: no cover - abstract
        raise NotImplementedError

    def commit(self, g: Geometry) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class RectTool(_DragTool):
    name = "rect"
    title = "Rectangle"
    shortcut = "B"
    hint = "Drag to draw a rectangle around a feature (Shift: square)."
    shape = Rect

    def deletable(self, a):
        return a.kind == "feature" and type(a.geometry) is self.shape

    def make(self, a, b, mods):
        (x0, y0), (x1, y1) = a, b
        if "shift" in mods:
            s = max(abs(x1 - x0), abs(y1 - y0))
            x1, y1 = x0 + math.copysign(s, x1 - x0 or 1), y0 + math.copysign(s, y1 - y0 or 1)
        return self.shape.from_corners(x0, y0, x1, y1)

    def commit(self, g):
        if g.w > 0 and g.h > 0:
            self._add("feature", g)


class EllipseTool(RectTool):
    name = "ellipse"
    title = "Ellipse"
    shortcut = "E"
    hint = "Drag to draw an ellipse (Shift: circle)."
    shape = Ellipse


class LineTool(_DragTool):
    """A straight segment of a given kind (measurement line, orientation vector, reference axis ...)."""
    kind = "measurement_line"

    def deletable(self, a):
        return a.kind == self.kind

    def make(self, a, b, mods):
        if "shift" in mods:                                # snap to 15° steps (on screen)
            sx, sy = self._scale()
            dx, dy = (b[0] - a[0]) * sx, (b[1] - a[1]) * sy
            r, t = math.hypot(dx, dy), math.atan2(dy, dx)
            t = round(t / math.radians(15)) * math.radians(15)
            b = (a[0] + r * math.cos(t) / sx, a[1] + r * math.sin(t) / sy)
        if a == b:
            return None
        return Segment([a, b])

    def commit(self, g):
        self._add(self.kind, g)


class MeasureTool(LineTool):
    name = "measure"
    title = "Measure / profile"
    shortcut = "M"
    hint = "Drag to measure a distance; the height profile along the line appears in the profile window."
    kind = "measurement_line"

    def commit(self, g):
        self._add(self.kind, g, label=None)


class OrientationTool(LineTool):
    name = "orientation"
    title = "Orientation vector"
    shortcut = "O"
    hint = ("Drag from tail to head across a feature: sets the feature's orientation (shown as an arrow from "
            "its centre). Away from a feature it adds a free vector. Shift snaps to 15° steps.")
    kind = "orientation_vector"

    def commit(self, g):
        store = self.session.store
        (x0, y0), (x1, y1) = g.points
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        parents = [a for a in store.hit_test(self.image_id, mx, my, 0.0, kinds=("feature",)) if a.geometry.closed]
        angle = angle_deg(x1 - x0, y1 - y0, self._scale())
        with store.transaction("Set orientation"):
            parent = parents[0] if parents else None
            if parent is not None:
                for old in store.children(parent.id):
                    if old.kind == "orientation_vector":
                        store.remove([old.id], cascade=False)
                store.update(parent.id, orientation=angle)
                self.session.select([parent.id])     # the arrow drawn from its centre shows the result
                return
            self._add(self.kind, g, label=self.session.current_label)


class HerringboneTool(LineTool):
    name = "herringbone"
    title = "Herringbone axis"
    shortcut = "W"
    hint = ("Drag along the herringbone ridges of one domain; ribbon alignment is measured against the nearest "
            "axis. Draw one per domain. Shift snaps to 15° steps.")
    kind = "herringbone_axis"

    def commit(self, g):
        self._add(self.kind, g, label=None)


class LatticeTool(LineTool):
    name = "lattice"
    title = "Lattice vector"
    shortcut = "L"
    hint = ("Drag a1, then a2, from the same atom (a2 snaps to a1's start). a1 of the first lattice is also the "
            "reference direction for orientations. Right-click → Lattice vectors… to type in known values. "
            "Shift snaps to 15° steps.")
    kind = "lattice_vector"

    def commit(self, g):
        store = self.session.store
        groups = lattice_groups(store.for_image(self.image_id, kinds=("lattice_vector",)))
        open_group = next((n for n, v in groups.items() if len(v) == 1), None)
        if open_group is not None:                       # second vector of a started lattice
            a1 = groups[open_group][0]
            (sx, sy), (ex, ey) = g.points
            o = a1.geometry.points[0]
            if math.hypot(sx - o[0], sy - o[1]) <= 0.35 * max(a1.geometry.length(), g.length()):
                g = Segment([tuple(o), (ex - sx + o[0], ey - sy + o[1])])      # from the same atom as a1
            self._add(self.kind, g, label=None, attrs={"lattice": open_group, "role": "a2"})
            return
        k = 1
        while f"lattice {k}" in groups:
            k += 1
        self._add(self.kind, g, label=None, attrs={"lattice": f"lattice {k}", "role": "a1"})


def lattice_groups(anns) -> dict:
    """{lattice name: [a1, a2]} of lattice-vector annotations (rejected ones left out)."""
    from ..measurements.report import lattice_pairs
    return lattice_pairs([a for a in anns if a.status != "rejected"], statuses=("manual", "accepted", "proposed"))


class PointTool(Tool):
    name = "point"
    title = "Point"
    shortcut = "P"
    hint = "Click to mark a feature position."

    def deletable(self, a):
        return a.kind == "feature" and isinstance(a.geometry, Point)

    def press(self, x, y, button="left", mods=frozenset()):
        if button != "left" or not self.image_id:
            return False
        self._add("feature", Point(*self._clip(x, y)))
        return True


class _ClickPathTool(Tool):
    """Click vertices; double-click or Enter finishes, Backspace removes the last vertex, Esc cancels."""
    closed = True
    min_vertices = 3

    def __init__(self, session, tolerance=lambda: 4.0):
        super().__init__(session, tolerance)
        self.points: list[tuple[float, float]] = []
        self._cursor: Optional[tuple[float, float]] = None

    @property
    def busy(self) -> bool:
        return bool(self.points)

    def press(self, x, y, button="left", mods=frozenset()):
        if not self.image_id:
            return False
        if button == "right":
            if self.points:
                self.finish()
                return True
            return False
        if button != "left":
            return False
        p = self._clip(x, y)
        if self.closed and len(self.points) >= self.min_vertices:
            fx, fy = self.points[0]
            if math.hypot(p[0] - fx, p[1] - fy) <= self.tolerance():     # click on the first vertex closes
                self.finish()
                return True
        if not self.points or math.hypot(p[0] - self.points[-1][0], p[1] - self.points[-1][1]) > 1e-6:
            self.points.append(p)
        self._update()
        return True

    def move(self, x, y, mods=frozenset()):
        if not self.points:
            return False
        self._cursor = self._clip(x, y)
        self._update()
        return True

    def double_click(self, x, y, mods=frozenset()):
        if self.points:
            self.finish()
            return True
        return False

    def key(self, key, mods=frozenset()):
        if not self.points:
            return False
        if key in ("enter", "return"):
            self.finish()
        elif key == "backspace":
            self.points.pop()
            self._update()
        elif key == "escape":
            self.cancel()
        else:
            return False
        return True

    def cancel(self):
        self.points = []
        self._cursor = None
        super().cancel()

    def _update(self) -> None:
        pts = list(self.points) + ([self._cursor] if self._cursor else [])
        color = self.session.label_color(self.session.current_label)
        self.preview = [Preview(Polyline(pts), "draft", color)] if len(pts) >= 2 else \
            ([Preview(Point(*pts[0]), "draft", color)] if pts else [])

    def finish(self) -> None:
        pts = _dedupe(self.points)
        self.cancel()
        if len(pts) >= self.min_vertices:
            self.commit(np.asarray(pts, float))

    def commit(self, pts: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def _dedupe(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


class PolygonTool(_ClickPathTool):
    name = "polygon"
    title = "Polygon"
    shortcut = "G"
    hint = "Click the corners; click the first point, double-click, right-click or press Enter to close."

    def deletable(self, a):
        return a.kind == "feature" and isinstance(a.geometry, Polygon)

    def commit(self, pts):
        g = Polygon(pts)
        if not g.is_degenerate():
            self._add("feature", g)


class CentreLineTool(_ClickPathTool):
    name = "centre_line"
    title = "Centre line"
    shortcut = "C"
    hint = "Click along the centre of a ribbon; double-click, right-click or Enter to finish."

    def deletable(self, a):
        return a.kind == "centre_line"
    closed = False
    min_vertices = 2

    def commit(self, pts):
        g = Segment(pts) if len(pts) == 2 else Polyline(pts)
        if not g.is_degenerate():
            self._add("centre_line", g)


class StepEdgeTool(_ClickPathTool):
    name = "step_edge"
    title = "Step edge"
    shortcut = "K"
    hint = "Click along a terrace step edge; double-click, right-click or Enter to finish."
    closed = False
    min_vertices = 2

    def deletable(self, a):
        return a.kind == "step_edge"

    def commit(self, pts):
        g = Segment(pts) if len(pts) == 2 else Polyline(pts)
        if not g.is_degenerate():
            self._add("step_edge", g, label=None)


class RegionTool(_ClickPathTool):
    name = "region"
    title = "Analysis region"
    shortcut = "J"
    hint = ("Click the corners of an area for the surface analyses (a domain, a terrace, or label it "
            "'exclude'); close like a polygon.")

    def deletable(self, a):
        return a.kind == "region"

    def commit(self, pts):
        g = Polygon(pts)
        if not g.is_degenerate():
            self._add("region", g)


class PaintTool(Tool):
    """Paints whole pixels into a mask feature, which keeps the exact shape (holes too)."""
    name = "paint"
    title = "Paint mask"
    shortcut = "D"
    hint = ("Drag to paint pixels: into the selected mask, the mask you start on, or a new one. Right-drag "
            "or Ctrl+drag erases. [ and ] change the brush size.")
    min_radius, max_radius = 0.5, 200.0

    def __init__(self, session, tolerance=lambda: 4.0):
        super().__init__(session, tolerance)
        self.radius = 3.0                          # brush radius, image pixels along x
        self._work: Optional[np.ndarray] = None    # the whole image's pixels while a stroke is drawn
        self._target: Optional[Annotation] = None
        self._erase = False
        self._last: Optional[tuple[float, float]] = None

    @property
    def busy(self) -> bool:
        return self._work is not None

    def deletable(self, a):
        return a.kind == "feature" and isinstance(a.geometry, Mask)

    def _masks_at(self, x: float, y: float) -> list[Annotation]:
        return [a for a in self.session.store.hit_test(self.image_id, x, y, 0.0, kinds=("feature",))
                if isinstance(a.geometry, Mask)]

    def _pick_target(self, x: float, y: float) -> Optional[Annotation]:
        sel = [a for a in self.session.selected() if a.image_id == self.image_id and self.deletable(a)]
        if len(sel) == 1:
            return sel[0]
        under = self._masks_at(x, y)
        return under[0] if under else None

    def _dab(self, x: float, y: float) -> None:
        """Paint (or erase) a round dab at (x, y) and the stroke from the last one."""
        import cv2
        sx, sy = self._scale()
        rx, ry = self.radius, self.radius * sx / sy                 # round on screen
        layer = np.zeros(self._work.shape, np.uint8)
        a, b = self._last or (x, y), (x, y)
        n = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / max(min(rx, ry) * 0.5, 0.5))))
        for k in range(n + 1):
            px, py = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
            centre = (int(round((px - 0.5) * 16)), int(round((py - 0.5) * 16)))   # pixel centres, 4 bits
            cv2.ellipse(layer, centre, (int(round(rx * 16)), int(round(ry * 16))), 0, 0, 360, 1, -1,
                        lineType=cv2.LINE_8, shift=4)
        if rx < 1 and ry < 1:                                       # at least the pixel under the cursor
            h, w = layer.shape
            c, r = int(math.floor(x)), int(math.floor(y))
            if 0 <= r < h and 0 <= c < w:
                layer[r, c] = 1
        if self._erase:
            self._work &= ~layer.astype(bool)
        else:
            self._work |= layer.astype(bool)
        self._last = (x, y)

    def _brush(self, x: float, y: float) -> Preview:
        sx, sy = self._scale()
        rx, ry = max(self.radius, 0.5), max(self.radius, 0.5) * sx / sy
        return Preview(Ellipse(x - rx, y - ry, 2 * rx, 2 * ry), "band")

    def press(self, x, y, button="left", mods=frozenset()):
        shape = self._shape()
        if button not in ("left", "right") or not self.image_id or shape is None:
            return False
        self._erase = button == "right" or "ctrl" in mods
        self._target = self._pick_target(x, y)
        if self._erase and self._target is None:
            return True
        self._work = self._target.geometry.mask(shape) if self._target is not None else np.zeros(shape, bool)
        self._last = None
        self._dab(x, y)
        self._update(x, y)
        return True

    def move(self, x, y, mods=frozenset()):
        if self._work is None:
            self.preview = [self._brush(x, y)] if self.image_id else []
            return True
        self._dab(x, y)
        self._update(x, y)
        return True

    def _update(self, x: float, y: float) -> None:
        label = self._target.label if self._target is not None else self.session.current_label
        self.preview = [Preview(Mask(0, 0, self._work), "mask", self.session.label_color(label)),
                        self._brush(x, y)]

    def release(self, x, y, button="left", mods=frozenset()):
        if self._work is None:
            return False
        g, target = Mask(0, 0, self._work), self._target
        self._work, self._target, self._last = None, None, None
        self.preview = [self._brush(x, y)]
        store = self.session.store
        if target is not None:
            if g.is_degenerate():
                store.remove([target.id], description="Erase mask")
            elif g != target.geometry:
                store.update(target.id, "Erase mask" if self._erase else "Paint mask", geometry=g)
            self.session.select([target.id])
        elif not g.is_degenerate():
            self._add("feature", g)
        return True

    def key(self, key, mods=frozenset()):
        if key in ("[", "]"):
            f = 1.25 if key == "]" else 0.8
            self.radius = float(min(max(self.radius * f, self.min_radius), self.max_radius))
            if self.preview:
                last = self.preview[-1].geometry
                cx, cy = last.centroid()
                self.preview[-1] = self._brush(cx, cy)
            return True
        return super().key(key, mods)

    def cancel(self):
        self._work, self._target, self._last = None, None, None
        super().cancel()


class PanTool(Tool):
    name = "pan"
    title = "Pan / zoom"
    shortcut = "H"
    hint = "Drag to pan, wheel to zoom. (Middle mouse button pans with any tool.)"
    cursor = "open_hand"


TOOLS: dict[str, type[Tool]] = {t.name: t for t in (SelectTool, PanTool, RectTool, EllipseTool, PolygonTool, PointTool,
                                                     CentreLineTool, MeasureTool, OrientationTool,
                                                     HerringboneTool, LatticeTool, StepEdgeTool, RegionTool,
                                                     PaintTool)}
