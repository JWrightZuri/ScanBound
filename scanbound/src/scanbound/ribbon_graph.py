"""
Ribbon graphs: the skeleton of an outline as a graph, split into separate ribbons.

A nanoribbon annotated as a polygon (or a tangle of fused ribbons annotated as one polygon) is
rasterised, thinned to a 1 px skeleton and turned into a graph: nodes are the free ends and the
junctions, edges are the skeleton branches between them.  Everything that measures ribbons builds
on this one graph:

* **length** of a ribbon = path length of its centre line, extended at each free end by the local
  half width (the skeleton stops half a width short of the outline);
* **width** = twice the mean distance from the centre line to the outline;
* **splitting** into separate ribbons: a ribbon carries straight on through a junction when two of
  the branches meeting there continue each other within ``straight_tolerance``; the others end
  there.  A sharp bend inside one branch (a turn of more than ``kink_angle`` over about two ribbon
  widths) also ends the ribbons on either side;
* **connections** between the ribbons (:func:`ribbon_connections`), from the split ribbons or from
  centre lines a person drew:

  ``branch``    a ribbon ends on the side of another one that continues through (T / Y shape)
  ``fusion``    ribbons end on each other at a sharp angle and none continues (V / L shape, kinks)
  ``crossing``  two ribbons pass through the same point (X shape)

No Qt, no global state.  All coordinates are scene coordinates (pixel (i, j) covers x in [j, j+1]).

    g = skeleton_graph(polygon_points)
    ribbons = split_ribbons(g)
    junctions = ribbon_connections(ribbons)
    table = ribbons_frame(ribbons, junctions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy import ndimage

__all__ = ["mask_ribbons", "mask_outlines", "GraphNode", "GraphEdge", "SkeletonGraph", "RibbonParams", "RibbonPath", "Junction",
           "skeleton_graph", "graph_from_mask", "split_ribbons", "ribbon_connections", "ribbons_from_polyline",
           "polygon_ribbons", "axial_angle", "JUNCTION_KINDS"]

JUNCTION_KINDS = ("branch", "fusion", "crossing")
_N8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class RibbonParams:
    """Settings of the skeleton graph and of the splitting.  Lengths are in ribbon widths."""
    kink_angle: float = 35.0          # a bend sharper than this (degrees) splits a ribbon
    straight_tolerance: float = 35.0  # branches within this of a straight line continue one ribbon
    spur_length: float = 1.0          # side branches shorter than this many widths are outline noise
    merge_distance: float = 1.0       # junctions closer than this many widths are one junction
    direction_length: float = 2.0     # directions at junctions / bends are measured over this many widths
    extend_ends: bool = True          # add the half width at free ends (skeleton stops short of the tip)
    min_pixels_across: float = 8.0    # rasterise so a ribbon is at least this many pixels wide

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# The graph
# =============================================================================

@dataclass
class GraphNode:
    id: int
    xy: tuple[float, float]
    edges: list = field(default_factory=list)      # ids of the incident edges (an edge twice for a loop)

    @property
    def degree(self) -> int:
        return len(self.edges)


@dataclass
class GraphEdge:
    id: int
    a: int                       # node at points[0] (-1: closed loop without nodes)
    b: int                       # node at points[-1]
    points: np.ndarray           # (N, 2) scene coordinates along the centre line
    half_width: np.ndarray       # (N,) distance to the outline at each point (scene px)

    @property
    def length(self) -> float:
        return _path_length(self.points)


@dataclass
class SkeletonGraph:
    nodes: dict                  # id -> GraphNode
    edges: dict                  # id -> GraphEdge
    width: float                 # mean width of the outline along its skeleton (scene px)

    @property
    def total_length(self) -> float:
        return float(sum(e.length for e in self.edges.values()))

    def junctions(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.degree >= 3]

    def ends(self) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.degree == 1]


def _path_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(np.asarray(p, float), axis=0).T).sum())


def skeleton_graph(polygon: np.ndarray, params: RibbonParams = RibbonParams(),
                   holes: Sequence[np.ndarray] = ()) -> Optional[SkeletonGraph]:
    """Skeleton graph of a polygon outline ((N, 2) scene coordinates, minus ``holes``), or None if it has no area."""
    pts = np.asarray(polygon, float).reshape(-1, 2)
    if len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perim = float(np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T).sum())
    if area <= 0 or perim <= 0:
        return None
    width_est = max(2.0 * area / perim, 1e-3)                # exact for a long rectangle
    x0, y0 = x.min(), y.min()
    span = max(x.max() - x0, y.max() - y0, 1.0)
    scale = max(1.0, params.min_pixels_across / width_est)
    scale = min(scale, max(1.0, 1600.0 / span), 16.0)        # keep the raster small
    pad = 2
    w = int(math.ceil((x.max() - x0) * scale)) + 2 * pad + 1
    h = int(math.ceil((y.max() - y0) * scale)) + 2 * pad + 1
    mask = np.zeros((h, w), np.uint8)

    # raster pixel (i, j) is centred on scene (x0 + (j - pad + 0.5) / scale, ...)
    def raster(p: np.ndarray) -> np.ndarray:
        return np.round(((np.asarray(p, float).reshape(-1, 2) - [x0, y0]) * scale + pad - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(mask, [raster(pts)], 1, lineType=cv2.LINE_8, shift=4)
    for hole in holes:
        if len(hole) >= 3:
            cv2.fillPoly(mask, [raster(hole)], 0, lineType=cv2.LINE_8, shift=4)
    if not mask.any():
        return None
    return graph_from_mask(mask.astype(bool), params, scale=scale, offset=(x0 - (pad - 0.5) / scale,
                                                                          y0 - (pad - 0.5) / scale))


def graph_from_mask(mask: np.ndarray, params: RibbonParams = RibbonParams(), scale: float = 1.0,
                    offset: tuple[float, float] = (0.5, 0.5)) -> Optional[SkeletonGraph]:
    """
    Skeleton graph of a bool mask (one object).  Raster pixel (i, j) maps to scene
    ``offset + (j, i) / scale``; the default maps a full-image mask to pixel centres.
    """
    from skimage.morphology import skeletonize
    mask = np.asarray(mask, bool)
    skel = skeletonize(mask)
    if not skel.any():
        return None
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    nodes, edges = _trace(skel, dist)
    off = np.asarray(offset, float)
    for e in edges.values():
        e.points = e.points / scale + off
        e.half_width = e.half_width / scale
    for n in nodes.values():
        n.xy = (n.xy[0] / scale + off[0], n.xy[1] / scale + off[1])
    sk_d = dist[skel]
    width = max(2.0 * float(sk_d.mean()) - 1.0, 1.0) / scale
    g = SkeletonGraph(nodes, edges, width)
    _simplify(g, params)
    return g


def _trace(skel: np.ndarray, dist: np.ndarray) -> tuple[dict, dict]:
    """Nodes (ends, junction clusters) and edges (pixel paths) of a skeleton, raster coordinates."""
    count = cv2.filter2D(skel.astype(np.float32), -1, _N8, borderType=cv2.BORDER_CONSTANT)
    junction = skel & (count >= 3)
    jlab, nj = ndimage.label(junction, structure=np.ones((3, 3), int))
    nodes: dict[int, GraphNode] = {}
    for k, c in enumerate(ndimage.center_of_mass(junction, jlab, range(1, nj + 1)), start=1):
        nodes[k] = GraphNode(k, (float(c[1]), float(c[0])))
    pieces = skel & ~junction
    plab, npc = ndimage.label(pieces, structure=np.ones((3, 3), int))
    H, W = skel.shape
    edges: dict[int, GraphEdge] = {}
    next_node = nj + 1
    for i, sl in enumerate(ndimage.find_objects(plab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(plab[sl] == i)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        path = _order_path(ys, xs)
        # junction clusters touching each end
        ends = []
        for (py, px) in (path[0], path[-1]):
            y0, y1, x0, x1 = max(py - 1, 0), min(py + 2, H), max(px - 1, 0), min(px + 2, W)
            found = np.unique(jlab[y0:y1, x0:x1])
            ends.append([int(v) for v in found if v])
        loop = len(path) > 2 and not ends[0] and not ends[1] and max(abs(path[0][0] - path[-1][0]),
                                                                      abs(path[0][1] - path[-1][1])) <= 1
        node_ids: list[int] = []
        for k, (touch, (py, px)) in enumerate(zip(ends, (path[0], path[-1]))):
            if len(path) == 1 and k == 1 and len(ends[0]) > 1:
                touch = [t for t in touch if t != node_ids[0]] or touch       # single pixel between two junctions
            if touch:
                node_ids.append(touch[0])
            elif loop:
                node_ids.append(-1)
            else:
                nodes[next_node] = GraphNode(next_node, (float(px), float(py)))
                node_ids.append(next_node)
                next_node += 1
        pts = _smooth(np.array([(px, py) for py, px in path], float), loop)
        hw = np.array([dist[py, px] for py, px in path], float)
        a, b = node_ids
        if a > 0 and pts.size and len(ends[0]):
            pts = np.vstack([nodes[a].xy, pts])
            hw = np.concatenate([[hw[0]], hw])
        if b > 0 and len(ends[1]):
            pts = np.vstack([pts, nodes[b].xy])
            hw = np.concatenate([hw, [hw[-1]]])
        e = GraphEdge(len(edges) + 1, a, b, pts, hw)
        edges[e.id] = e
        for n in (a, b):
            if n > 0:
                nodes[n].edges.append(e.id)
    # junction clusters that touch each other directly (no edge pixels between) are already one cluster;
    # a lone junction cluster (no edges, e.g. a 2x2 blob) becomes an isolated node
    return nodes, edges


def _smooth(pts: np.ndarray, loop: bool, sigma: float = 2.0) -> np.ndarray:
    """Smooth the pixel staircase of a skeleton path (8-connected steps overestimate oblique lengths)."""
    if len(pts) < 5:
        return pts
    out = ndimage.gaussian_filter1d(pts, sigma, axis=0, mode="wrap" if loop else "nearest")
    if not loop:
        out[0], out[-1] = pts[0], pts[-1]
    return out


def _order_path(ys: np.ndarray, xs: np.ndarray) -> list[tuple[int, int]]:
    """Order the pixels of a simple 8-connected path (or loop) from one end to the other."""
    pix = list(zip(ys.tolist(), xs.tolist()))
    if len(pix) <= 2:
        return pix
    index = {p: k for k, p in enumerate(pix)}
    nbrs: list[list[int]] = [[] for _ in pix]
    for k, (y, x) in enumerate(pix):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in index:
                    nbrs[k].append(index[(y + dy, x + dx)])
    start = next((k for k in range(len(pix)) if len(nbrs[k]) == 1), 0)
    order = [start]
    seen = {start}
    cur = start
    while True:
        cand = [n for n in nbrs[cur] if n not in seen]
        if not cand:
            break
        # prefer 4-neighbours so diagonal shortcuts do not skip a pixel
        cy, cx = pix[cur]
        cand.sort(key=lambda n: abs(pix[n][0] - cy) + abs(pix[n][1] - cx))
        cur = cand[0]
        order.append(cur)
        seen.add(cur)
    if len(order) < len(pix):          # a small side blob the walk missed: append in distance order
        rest = [k for k in range(len(pix)) if k not in seen]
        order.extend(rest)
    return [pix[k] for k in order]


# ---- graph clean-up --------------------------------------------------------------------------

def _other(e: GraphEdge, n: int) -> int:
    return e.b if e.a == n else e.a


def _remove_edge(g: SkeletonGraph, eid: int) -> None:
    e = g.edges.pop(eid)
    for n in (e.a, e.b):
        if n in g.nodes and eid in g.nodes[n].edges:
            g.nodes[n].edges.remove(eid)


def _join_through(g: SkeletonGraph, nid: int) -> bool:
    """Merge the two edges of a degree-2 node into one edge."""
    node = g.nodes[nid]
    if node.degree != 2 or node.edges[0] == node.edges[1]:
        return False
    e1, e2 = (g.edges[i] for i in node.edges)
    p1, h1 = (e1.points, e1.half_width) if e1.b == nid else (e1.points[::-1], e1.half_width[::-1])
    p2, h2 = (e2.points, e2.half_width) if e2.a == nid else (e2.points[::-1], e2.half_width[::-1])
    a = e1.a if e1.b == nid else e1.b
    b = e2.b if e2.a == nid else e2.a
    _remove_edge(g, e1.id)
    _remove_edge(g, e2.id)
    del g.nodes[nid]
    new = GraphEdge(max(g.edges, default=0) + 1, a, b, np.vstack([p1, p2[1:]]), np.concatenate([h1, h2[1:]]))
    if a == b:                                    # became a loop through one node
        pass
    g.edges[new.id] = new
    for n in (a, b):
        g.nodes[n].edges.append(new.id)
    return True


def _simplify(g: SkeletonGraph, p: RibbonParams) -> None:
    w = g.width
    changed = True
    while changed:
        changed = False
        # 1. spurs: short branches from a junction to a free end are bumps of the outline
        for eid in list(g.edges):
            e = g.edges.get(eid)
            if e is None or e.a < 0:
                continue
            da, db = g.nodes[e.a].degree, g.nodes[e.b].degree
            if {da, db} == {1} or not ((da == 1 and db >= 3) or (db == 1 and da >= 3)):
                continue
            junction = e.a if da >= 3 else e.b
            hw = float(np.median(e.half_width))
            if e.length < max(p.spur_length * w, 2.0 * hw):
                end = e.b if junction == e.a else e.a
                _remove_edge(g, eid)
                del g.nodes[end]
                changed = True
        # 2. junctions joined by a short edge are one junction (an X splits into two Y's)
        for eid in list(g.edges):
            e = g.edges.get(eid)
            if e is None or e.a < 0 or e.a == e.b:
                continue
            if g.nodes[e.a].degree >= 3 and g.nodes[e.b].degree >= 3 and e.length < p.merge_distance * w:
                keep, drop = e.a, e.b
                _remove_edge(g, eid)
                kx, ky = g.nodes[keep].xy
                dx_, dy_ = g.nodes[drop].xy
                g.nodes[keep].xy = ((kx + dx_) / 2, (ky + dy_) / 2)
                for other in list(g.nodes[drop].edges):
                    oe = g.edges[other]
                    if oe.a == drop:
                        oe.a = keep
                        oe.points = np.vstack([g.nodes[keep].xy, oe.points])
                        oe.half_width = np.concatenate([[oe.half_width[0]], oe.half_width])
                    if oe.b == drop:
                        oe.b = keep
                        oe.points = np.vstack([oe.points, g.nodes[keep].xy])
                        oe.half_width = np.concatenate([oe.half_width, [oe.half_width[-1]]])
                    g.nodes[keep].edges.append(other)
                del g.nodes[drop]
                changed = True
        # 3. nodes left with two edges are not junctions any more
        for nid in list(g.nodes):
            if nid in g.nodes and g.nodes[nid].degree == 2 and _join_through(g, nid):
                changed = True
        for nid in [n for n, node in g.nodes.items() if node.degree == 0 and len(g.nodes) > 1]:
            del g.nodes[nid]


# =============================================================================
# Ribbons
# =============================================================================

@dataclass
class RibbonPath:
    """One ribbon: a centre line with its width; ends know what they are attached to."""
    points: np.ndarray                       # (N, 2) scene coordinates
    width: float                             # scene px
    source: str = ""                         # the annotation it came from
    end_extension: tuple[float, float] = (0.0, 0.0)   # length added at the start / end (free ends)
    closed: bool = False

    @property
    def contour_length(self) -> float:
        return _path_length(self.points) + sum(self.end_extension)

    @property
    def end_to_end(self) -> float:
        return float(np.hypot(*(self.points[-1] - self.points[0]))) + sum(self.end_extension)

    @property
    def axis_angle(self) -> float:
        """Principal direction of the centre line, degrees in [0, 180) counter-clockwise from +x."""
        return axial_angle(self.points)


def axial_angle(points: np.ndarray) -> float:
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return float("nan")
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    if L.sum() <= 0:
        return float("nan")
    # length-weighted mean of the doubled segment angles: the axis of a curved line, not its chord
    t = np.arctan2(-seg[:, 1], seg[:, 0])
    c, s = (L * np.cos(2 * t)).sum(), (L * np.sin(2 * t)).sum()
    if math.hypot(c, s) < 1e-9 * L.sum():
        d = pts[-1] - pts[0]
        return float(np.degrees(np.arctan2(-d[1], d[0])) % 180.0)
    return float(np.degrees(0.5 * np.arctan2(s, c)) % 180.0)


def _point_at(points: np.ndarray, dist: float) -> np.ndarray:
    """Point at arc length ``dist`` from points[0] (clamped)."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = min(max(dist, 0.0), cum[-1])
    return np.array([np.interp(d, cum, points[:, 0]), np.interp(d, cum, points[:, 1])])


def _end_direction(points: np.ndarray, reach: float) -> np.ndarray:
    """Unit vector from points[0] into the line, over ``reach`` (at most half of it)."""
    L = _path_length(points)
    q = _point_at(points, min(reach, 0.5 * L) if L > 0 else 0)
    v = q - points[0]
    n = np.hypot(*v)
    return v / n if n > 0 else np.zeros(2)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _kinks(points: np.ndarray, reach: float, kink_angle: float) -> list[int]:
    """Indices of sharp bends: the direction turns by more than kink_angle within ``reach``."""
    if len(points) < 3:
        return []
    seg = np.hypot(*np.diff(points, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    if L < 2 * reach:
        return []
    inner = (cum >= reach) & (cum <= L - reach)
    before = np.stack([np.interp(cum - reach, cum, points[:, 0]), np.interp(cum - reach, cum, points[:, 1])], 1)
    after = np.stack([np.interp(cum + reach, cum, points[:, 0]), np.interp(cum + reach, cum, points[:, 1])], 1)
    u = points - before
    v = after - points
    nu = np.hypot(u[:, 0], u[:, 1])
    nv = np.hypot(v[:, 0], v[:, 1])
    ok = inner & (nu > 0) & (nv > 0)
    cosang = np.where(ok, (u * v).sum(axis=1) / np.where(ok, nu * nv, 1.0), 1.0)
    turns = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    out: list[int] = []
    for k in np.argsort(-turns):
        if turns[k] <= kink_angle:
            break
        if all(abs(cum[k] - cum[j]) > reach for j in out):
            out.append(int(k))
    return sorted(out)


def ribbons_from_polyline(points: np.ndarray, width: float, params: RibbonParams = RibbonParams(),
                          source: str = "") -> list[RibbonPath]:
    """A drawn centre line, split at its sharp bends (no width information: ends are not extended)."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 2:
        return []
    dense = _densify(pts, max(width / 4.0, 0.5))
    cuts = _kinks(dense, params.direction_length * width / 2.0, params.kink_angle)
    out = []
    bounds = [0] + cuts + [len(dense) - 1]
    for i, j in zip(bounds[:-1], bounds[1:]):
        if j > i:
            out.append(RibbonPath(dense[i:j + 1], width, source))
    return out


def _densify(pts: np.ndarray, step: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
        for t in np.arange(1, n + 1) / n:
            out.append(a + (b - a) * t)
    return np.asarray(out)


def split_ribbons(g: SkeletonGraph, params: RibbonParams = RibbonParams(), source: str = "") -> list[RibbonPath]:
    """The separate ribbons of a skeleton graph (see the module docstring)."""
    if g is None or not g.edges:
        return []
    w = g.width
    reach = params.direction_length * w
    # 1. cut every edge at its sharp bends -> segments with an id per end
    segs: list[dict] = []
    for e in g.edges.values():
        cuts = _kinks(e.points, reach / 2.0, params.kink_angle)
        bounds = [0] + cuts + [len(e.points) - 1]
        for k, (i, j) in enumerate(zip(bounds[:-1], bounds[1:])):
            segs.append({"pts": e.points[i:j + 1], "hw": e.half_width[i:j + 1],
                         "a": e.a if k == 0 else ("kink", e.id, i),
                         "b": e.b if k == len(bounds) - 2 else ("kink", e.id, j),
                         "loop": e.a < 0})
    # 2. at each junction, pair the branch ends that continue each other
    link: dict[tuple[int, int], tuple[int, int]] = {}
    ends_at: dict = {}
    for si, s in enumerate(segs):
        for side, node in ((0, s["a"]), (1, s["b"])):
            if isinstance(node, int) and node < 0:
                continue
            ends_at.setdefault(node, []).append((si, side))
    for node, ends in ends_at.items():
        if not isinstance(node, int) or len(ends) < 3:
            continue          # kinks and free ends stop the ribbon; degree-2 nodes do not exist
        dirs = {}
        for si, side in ends:
            p = segs[si]["pts"] if side == 0 else segs[si]["pts"][::-1]
            dirs[(si, side)] = _end_direction(p, reach)
        free = list(ends)
        while len(free) >= 2:
            best = None
            for i in range(len(free)):
                for j in range(i + 1, len(free)):
                    if free[i][0] == free[j][0]:
                        continue                          # a loop's two ends
                    dev = 180.0 - _angle_between(dirs[free[i]], dirs[free[j]])
                    if best is None or dev < best[0]:
                        best = (dev, i, j)
            if best is None or best[0] > params.straight_tolerance:
                break
            _, i, j = best
            link[free[i]] = free[j]
            link[free[j]] = free[i]
            free = [f for k, f in enumerate(free) if k not in (i, j)]
    # 3. chain the linked segments into ribbons
    used = [False] * len(segs)
    out: list[RibbonPath] = []

    def walk(si: int, side: int) -> RibbonPath:
        pts_l: list[np.ndarray] = []
        hw_l: list[np.ndarray] = []
        start_end = (si, side)
        cur = (si, side)
        closed = False
        while True:
            s_i, s_side = cur
            used[s_i] = True
            s = segs[s_i]
            p, h = (s["pts"], s["hw"]) if s_side == 0 else (s["pts"][::-1], s["hw"][::-1])
            pts_l.append(p if not pts_l else p[1:])
            hw_l.append(h if not hw_l else h[1:])
            far = (s_i, 1 - s_side)
            nxt = link.get(far)
            if nxt is None:
                break
            if used[nxt[0]]:
                closed = nxt == start_end
                break
            cur = nxt
        pts = np.vstack(pts_l)
        hw = np.concatenate(hw_l)
        ext = (0.0, 0.0)
        if params.extend_ends and not closed:
            first_free = _is_free(segs, start_end, ends_at)
            last_free = _is_free(segs, far, ends_at)
            ext = (float(hw[0]) if first_free else 0.0, float(hw[-1]) if last_free else 0.0)
        width = max(2.0 * float(hw.mean()), 1e-6) if len(hw) else w
        return RibbonPath(pts, width, source, ext, closed)

    for si in range(len(segs)):                 # start at ends that are not linked
        for side in (0, 1):
            if not used[si] and (si, side) not in link:
                out.append(walk(si, side))
    for si in range(len(segs)):                 # what is left are closed loops
        if not used[si]:
            out.append(walk(si, 0))
    return [r for r in out if len(r.points) >= 2]


def _is_free(segs: list, end: tuple[int, int], ends_at: dict) -> bool:
    si, side = end
    node = segs[si]["a"] if side == 0 else segs[si]["b"]
    if not isinstance(node, int):
        return False                 # a kink
    return len(ends_at.get(node, [])) == 1


def polygon_ribbons(polygon: np.ndarray, params: RibbonParams = RibbonParams(), source: str = ""
                    ) -> tuple[Optional[SkeletonGraph], list[RibbonPath]]:
    g = skeleton_graph(polygon, params)
    return g, (split_ribbons(g, params, source) if g is not None else [])


def mask_outlines(bits: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)
                  ) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    """
    (outer boundary, holes) of the largest connected piece of a painted mask, in scene
    coordinates (pixel centres of the boundary pixels; ``offset`` is the top-left pixel corner).
    """
    m = np.ascontiguousarray(np.asarray(bits, bool), np.uint8)
    if not m.any():
        return None, []
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, []
    off = np.array([float(offset[0]) + 0.5, float(offset[1]) + 0.5])
    outers = [i for i in range(len(cs)) if hier[0][i][3] < 0]
    k = max(outers, key=lambda i: (cv2.contourArea(cs[i]), len(cs[i])))
    holes = [cs[i][:, 0, :].astype(float) + off for i in range(len(cs)) if hier[0][i][3] == k and len(cs[i]) >= 3]
    return cs[k][:, 0, :].astype(float) + off, holes


def mask_ribbons(bits: np.ndarray, offset: tuple[float, float] = (0.0, 0.0),
                 params: RibbonParams = RibbonParams(), source: str = ""
                 ) -> tuple[Optional[SkeletonGraph], list[RibbonPath]]:
    """
    Like :func:`polygon_ribbons` for a painted mask: ``bits`` (bool, holes kept) with its top-left
    pixel corner at scene ``offset``.  The largest connected piece is traced along its pixel
    boundary (holes included), so its skeleton is as smooth as that of a polygon outline.
    """
    outer, holes = mask_outlines(bits, offset)
    if outer is None:
        return None, []
    if len(outer) < 3 or cv2.contourArea(outer.astype(np.float32)) <= 0:   # one pixel wide: trace the pixels
        g = graph_from_mask(np.asarray(bits, bool), params,
                            offset=(float(offset[0]) + 0.5, float(offset[1]) + 0.5))
    else:
        g = skeleton_graph(outer, params, holes)
    return g, (split_ribbons(g, params, source) if g is not None else [])


# =============================================================================
# Connections between ribbons
# =============================================================================

@dataclass
class Junction:
    kind: str                        # "branch" / "fusion" / "crossing"
    xy: tuple[float, float]
    ends: list                       # (ribbon index, 0 = start / 1 = end) of the ribbons ending here
    sides: list                      # ribbon indices that pass through here
    angle: float                     # smallest angle between the ribbons meeting here (degrees, 0-90)
    source: str = ""


def _closest_on(points: np.ndarray, q: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(distance, arc position, closest point) of q to a polyline."""
    a = points[:-1]
    d = np.diff(points, axis=0)
    L2 = (d ** 2).sum(axis=1)
    t = np.clip(((q - a) * d).sum(axis=1) / np.where(L2 > 0, L2, 1), 0, 1)
    c = a + d * t[:, None]
    dist = np.hypot(*(c - q).T)
    k = int(np.argmin(dist))
    seg = np.sqrt(L2)
    arc = float(seg[:k].sum() + seg[k] * t[k])
    return float(dist[k]), arc, c[k]


def _dir_at(points: np.ndarray, arc: float, reach: float) -> np.ndarray:
    L = _path_length(points)
    a = _point_at(points, arc - reach)
    b = _point_at(points, arc + reach)
    v = b - a
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def _axial(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two lines (0-90°)."""
    a = _angle_between(u, v)
    return min(a, 180.0 - a)


def ribbon_connections(ribbons: Sequence[RibbonPath], tolerance: Optional[float] = None,
                       reach_widths: float = 2.0) -> list[Junction]:
    """
    Where the ribbons touch.  Ends closer than ``tolerance`` (default: 0.6 x the larger width, at
    least 1.5 px) are one junction; an end that touches another ribbon away from its ends makes
    a branch; ribbons whose interiors cross make a crossing.  Only ribbons from the same source
    annotation, or touching ones from different annotations, are compared.
    """
    n = len(ribbons)
    if n == 0:
        return []

    def tol(i: int, j: int) -> float:
        if tolerance is not None:
            return tolerance
        return max(1.5, 0.6 * max(ribbons[i].width, ribbons[j].width))

    boxes = np.array([[r.points[:, 0].min(), r.points[:, 1].min(), r.points[:, 0].max(), r.points[:, 1].max()]
                      for r in ribbons])
    ends = [(i, s) for i in range(n) for s in (0, 1) if not ribbons[i].closed]
    pos = {(i, s): ribbons[i].points[0 if s == 0 else -1] for i, s in ends}
    # 1. clusters of ends (union-find)
    parent = {e: e for e in ends}

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ea, eb = ends[a], ends[b]
            if ea[0] == eb[0]:
                continue
            if np.hypot(*(pos[ea] - pos[eb])) <= tol(ea[0], eb[0]):
                parent[find(ea)] = find(eb)
    clusters: dict = {}
    for e in ends:
        clusters.setdefault(find(e), []).append(e)
    out: list[Junction] = []
    for members in clusters.values():
        xy = np.mean([pos[e] for e in members], axis=0)
        # does an end of this cluster land on the side of another ribbon?
        sides = []
        member_ribbons = {i for i, _ in members}
        for j in range(n):
            if j in member_ribbons:
                continue
            t = tol(members[0][0], j)
            b = boxes[j]
            if xy[0] < b[0] - t or xy[0] > b[2] + t or xy[1] < b[1] - t or xy[1] > b[3] + t:
                continue
            d, arc, _ = _closest_on(ribbons[j].points, xy)
            L = _path_length(ribbons[j].points)
            if d <= t and (ribbons[j].closed or t < arc < L - t):
                sides.append((j, arc))
        if len(members) == 1 and not sides:
            continue                                             # a free end
        dirs = []
        for i, s in members:
            p = ribbons[i].points if s == 0 else ribbons[i].points[::-1]
            dirs.append(_end_direction(p, reach_widths * ribbons[i].width))
        for j, arc in sides:
            dirs.append(_dir_at(ribbons[j].points, arc, reach_widths * ribbons[j].width))
        angles = [_axial(dirs[a], dirs[b]) for a in range(len(dirs)) for b in range(a + 1, len(dirs))]
        kind = "branch" if sides else "fusion"
        src = ribbons[members[0][0]].source
        out.append(Junction(kind, (float(xy[0]), float(xy[1])), sorted(members), [j for j, _ in sides],
                            float(min(angles)) if angles else float("nan"), src))
    # 2. crossings: interiors that intersect (the split of a skeleton X leaves two through-ribbons)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = boxes[i], boxes[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            for hit, ai, aj in _intersections(ribbons[i].points, ribbons[j].points):
                ti, tj = tol(i, j), tol(i, j)
                Li, Lj = _path_length(ribbons[i].points), _path_length(ribbons[j].points)
                if not (ti < ai < Li - ti and tj < aj < Lj - tj):
                    continue                                     # at an end: handled above
                near = 1.5 * max(ribbons[i].width, ribbons[j].width)       # wiggles of one crossing
                if any(np.hypot(hit[0] - k.xy[0], hit[1] - k.xy[1]) <= (near if set(k.sides) == {i, j} else ti)
                       for k in out):
                    continue
                di = _dir_at(ribbons[i].points, ai, reach_widths * ribbons[i].width)
                dj = _dir_at(ribbons[j].points, aj, reach_widths * ribbons[j].width)
                out.append(Junction("crossing", (float(hit[0]), float(hit[1])), [], [i, j], _axial(di, dj),
                                    ribbons[i].source))
    return out


def _intersections(p: np.ndarray, q: np.ndarray) -> list[tuple[tuple[float, float], float, float]]:
    """Intersection points of two polylines with their arc positions on each (shared vertices count)."""
    out = []
    a0, a1 = p[:-1], p[1:]
    b0, b1 = q[:-1], q[1:]
    la = np.hypot(*(a1 - a0).T)
    lb = np.hypot(*(b1 - b0).T)
    ca = np.concatenate([[0], np.cumsum(la)])
    cb = np.concatenate([[0], np.cumsum(lb)])
    da = a1 - a0
    db = b1 - b0
    for k in range(len(a0)):
        r = da[k]
        den = r[0] * db[:, 1] - r[1] * db[:, 0]
        diff = b0 - a0[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (diff[:, 0] * db[:, 1] - diff[:, 1] * db[:, 0]) / den
            u = (diff[:, 0] * r[1] - diff[:, 1] * r[0]) / den
        hit = np.nonzero((den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))[0]
        for m in hit:
            xy = a0[k] + r * t[m]
            out.append(((float(xy[0]), float(xy[1])), float(ca[k] + la[k] * t[m]), float(cb[m] + lb[m] * u[m])))
    # merge hits at a shared vertex
    merged: list = []
    for h in out:
        if all(math.hypot(h[0][0] - m[0][0], h[0][1] - m[0][1]) > 1e-6 for m in merged):
            merged.append(h)
    return merged
