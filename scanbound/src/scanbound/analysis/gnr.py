"""
Skeleton analyses of any area ROI (polygon, rectangle, ellipse or painted mask, of any label):
segment lengths and widths, branching and fusion, alignment to the herringbone.

Both share one segment set per run (:func:`ribbon_set`): every shape in scope is turned into its
skeleton graph and split into separate segments at junctions and sharp bends
(:mod:`scanbound.ribbon_graph`), and centre lines drawn by a person replace the skeleton of the
shape they lie in.  So when the automatic split is wrong, draw (or correct) its centre lines and
the analyses use those.  (The internal names say "ribbon": the code was written for nanoribbons.)

skeleton    length / width / straightness of every segment, segments per shape, branch, fusion and
            crossing points, with histograms (bins set by the user) and maps of the split
alignment   angle of every segment to the Au(111) herringbone ridges next to it: marked by hand
            (herringbone axis annotations, each counting for the domain it lies in) or measured
            from the image, which is split into herringbone domains

The skeleton results are cached by shape and settings, so re-running with other bins or plots
is instant.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..herringbone import AU111_PERIOD_NM, axial_difference, herringbone_domains, herringbone_field
from ..ribbon_graph import (RibbonParams, RibbonPath, mask_ribbons, polygon_ribbons, ribbon_connections,
                            ribbons_from_polyline)
from .roi import AREA_TYPES, AnalysisContext, AnalysisResult, Param, register_analysis

SOURCES = ("Centre lines where drawn, else shapes", "Shapes (skeleton)", "Centre lines only")
JUNCTION_COLORS = {"branch": "#E8412C", "fusion": "#FFC233", "crossing": "#2EC4E6"}
HERRINGBONE_KIND = "herringbone_axis"

_GRAPH_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_GRAPH_CACHE_SIZE = 20000


def _cached_shape_ribbons(shape: dict, params: RibbonParams):
    """(width, skeleton length, segments) of an outline or a painted mask, cached by geometry and settings."""
    bits = shape.get("mask")
    if bits is not None:
        bits = np.asarray(bits, bool)
        ox, oy = shape.get("offset", (0.0, 0.0))
        off = (float(ox), float(oy))
        key: tuple = ("mask", bits.shape, np.packbits(bits).tobytes(), off, params)
    else:
        key = (np.asarray(shape["pts"], np.float64).tobytes(), params)
    hit = _GRAPH_CACHE.get(key)
    if hit is not None:
        _GRAPH_CACHE.move_to_end(key)
        return hit
    g, ribbons = mask_ribbons(bits, off, params) if bits is not None else polygon_ribbons(shape["pts"], params)
    value = (g.width if g is not None else float("nan"), g.total_length if g is not None else 0.0, ribbons)
    _GRAPH_CACHE[key] = value
    if len(_GRAPH_CACHE) > _GRAPH_CACHE_SIZE:
        _GRAPH_CACHE.popitem(last=False)
    return value


# =============================================================================
# The shared ribbon set
# =============================================================================

@dataclass
class RibbonSet:
    ribbons: list                      # RibbonPath, scene px (of their own image)
    table: pd.DataFrame                # one row per ribbon (aligned with ``ribbons``)
    shapes: pd.DataFrame               # one row per outline / free centre line
    junctions: pd.DataFrame            # one row per branch / fusion / crossing point
    notes: list = field(default_factory=list)


def ribbon_params(kink_angle=35.0, straight_tolerance=35.0, spur_length=1.0, extend_ends=True) -> RibbonParams:
    return RibbonParams(kink_angle=float(kink_angle), straight_tolerance=float(straight_tolerance),
                        spur_length=float(spur_length), merge_distance=1.0, extend_ends=bool(extend_ends))


def _inside_share(line: np.ndarray, poly: np.ndarray) -> float:
    c = poly.astype(np.float32).reshape(-1, 1, 2)
    inside = [cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in line]
    return float(np.mean(inside)) if inside else 0.0


def ribbon_set(ctx: AnalysisContext, params: RibbonParams = RibbonParams(), source: str = SOURCES[0],
               line_width: float = 0.0) -> RibbonSet:
    """Every segment of the shapes in scope (cached on the context per settings)."""
    cache = ctx.__dict__.setdefault("_ribbon_sets", {})
    key = (params, source, float(line_width))
    if key in cache:
        return cache[key]
    notes: list[str] = []
    area_shapes = [s for s in ctx.shapes if s["type"] in AREA_TYPES and s["type"] != "PointROI"]
    line_shapes = [s for s in ctx.shapes if s["type"] == "LineSegmentROI"]
    if source == SOURCES[1]:
        line_shapes = []
    if source == SOURCES[2]:
        area_shapes = []
    # which drawn centre lines lie in which outline
    owner: dict[int, int] = {}
    if area_shapes and line_shapes:
        by_file: dict = {}
        for i, s in enumerate(area_shapes):
            by_file.setdefault(s["file"], []).append(i)
        for j, l in enumerate(line_shapes):
            best, share = None, 0.5
            lb = (l["pts"][:, 0].min(), l["pts"][:, 1].min(), l["pts"][:, 0].max(), l["pts"][:, 1].max())
            for i in by_file.get(l["file"], []):
                p = area_shapes[i]["pts"]
                if lb[2] < p[:, 0].min() or lb[0] > p[:, 0].max() or lb[3] < p[:, 1].min() or lb[1] > p[:, 1].max():
                    continue
                sh = _inside_share(l["pts"], p)
                if sh > share:
                    best, share = i, sh
            if best is not None:
                owner[j] = best
    lines_of: dict[int, list[int]] = {}
    for j, i in owner.items():
        lines_of.setdefault(i, []).append(j)

    ribbons: list[RibbonPath] = []
    meta: list[dict] = []              # per ribbon: file, objectName, label, group
    shape_rows: list[dict] = []
    widths_px: list[float] = []
    for i, s in enumerate(area_shapes):
        width, skel_len, rs = _cached_shape_ribbons(s, params)
        drawn = lines_of.get(i, [])
        how = "skeleton"
        if drawn:
            how = "centre lines"
            w = width if math.isfinite(width) else 4.0
            rs = [r for j in drawn for r in ribbons_from_polyline(line_shapes[j]["pts"], w, params,
                                                                 line_shapes[j]["objectName"])]
        if math.isfinite(width):
            widths_px.append(width * ctx.scale(s["file"]))
        group = (s["file"], s["objectName"])
        for r in rs:
            ribbons.append(r)
            meta.append({"file": s["file"], "objectName": s["objectName"], "label": s["label"], "group": group})
        shape_rows.append({"file": s["file"], "objectName": s["objectName"], "label": s["label"], "type": s["type"],
                           "segments from": how, "_width_px": width, "_skeleton_px": skel_len})
    default_w = float(np.median(widths_px)) if widths_px else None
    free = [j for j in range(len(line_shapes)) if j not in owner]
    for j in free:
        l = line_shapes[j]
        sc = ctx.scale(l["file"])
        if line_width and line_width > 0:
            w = line_width / sc
        elif default_w:
            w = default_w / sc
        else:
            w = 4.0
        group = (l["file"], "__lines__")              # drawn lines of one image can join each other
        for r in ribbons_from_polyline(l["pts"], w, params, l["objectName"]):
            ribbons.append(r)
            meta.append({"file": l["file"], "objectName": l["objectName"], "label": l["label"], "group": group})
        shape_rows.append({"file": l["file"], "objectName": l["objectName"], "label": l["label"],
                           "type": l["type"], "segments from": "centre line", "_width_px": w,
                           "_skeleton_px": float(np.hypot(*np.diff(l["pts"], axis=0).T).sum())})
    if free and not (line_width and line_width > 0) and not default_w:
        notes.append("Centre lines have no width: 4 px is assumed for joining them (set 'Centre line width').")

    # connections, per group
    groups: dict = {}
    for k, m in enumerate(meta):
        groups.setdefault(m["group"], []).append(k)
    counts = {k: {"branches on it": 0, "branched off another": 0, "fusion points": 0, "crossings": 0}
              for k in range(len(ribbons))}
    jrows = []
    for group, members in groups.items():
        if len(members) < 2:
            continue
        sub = [ribbons[k] for k in members]
        for jn in ribbon_connections(sub):
            ends = [members[i] for i, _ in jn.ends]
            sides = [members[i] for i in jn.sides]
            if jn.kind == "branch":
                for k in ends:
                    counts[k]["branched off another"] += 1
                for k in sides:
                    counts[k]["branches on it"] += 1
            elif jn.kind == "fusion":
                for k in set(ends):
                    counts[k]["fusion points"] += 1
            else:
                for k in sides:
                    counts[k]["crossings"] += 1
            file = group[0]
            sc = ctx.scale(file)
            objs = sorted({meta[k]["objectName"] for k in ends + sides})
            jrows.append({"file": file, "objectName": objs[0] if len(objs) == 1 else ", ".join(objs),
                          "label": meta[(ends + sides)[0]]["label"], "kind": jn.kind, "x_px": jn.xy[0],
                          "y_px": jn.xy[1], "segments": len(set(ends + sides)), "ends": len(ends),
                          "angle (°)": jn.angle, "_ribbon_ids": tuple(sorted(set(ends + sides)))})
    # tables in display units
    rows = []
    per_obj_index: dict = {}
    for k, (r, m) in enumerate(zip(ribbons, meta)):
        sc = ctx.scale(m["file"])
        per_obj_index[(m["file"], m["objectName"])] = per_obj_index.get((m["file"], m["objectName"]), 0) + 1
        c = counts[k]
        L = r.contour_length * sc
        E = r.end_to_end * sc
        cx, cy = r.points.mean(axis=0)
        rows.append({"file": m["file"], "objectName": m["objectName"], "label": m["label"],
                     "segment": per_obj_index[(m["file"], m["objectName"])], "length": L, "width": r.width * sc,
                     "end_to_end": E, "straightness": E / L if L > 0 else np.nan, "angle (°)": r.axis_angle,
                     **c, "connections": sum(c.values()), "closed loop": r.closed, "cx_px": cx, "cy_px": cy})
    table = pd.DataFrame(rows, columns=["file", "objectName", "label", "segment", "length", "width", "end_to_end",
                                        "straightness", "angle (°)", "branches on it", "branched off another",
                                        "fusion points", "crossings", "connections", "closed loop", "cx_px", "cy_px"])
    junctions = pd.DataFrame(jrows, columns=["file", "objectName", "label", "kind", "x_px", "y_px", "segments",
                                             "ends", "angle (°)", "_ribbon_ids"])
    shapes = pd.DataFrame(shape_rows, columns=["file", "objectName", "label", "type", "segments from",
                                               "_width_px", "_skeleton_px"])
    if len(shapes):
        sc = shapes["file"].map(ctx.scale).astype(float)
        by_obj = table.groupby(["file", "objectName"])
        n = by_obj.size()
        tot = by_obj["length"].sum()
        key_idx = pd.MultiIndex.from_frame(shapes[["file", "objectName"]])
        shapes["segments"] = n.reindex(key_idx).fillna(0).astype(int).values
        shapes["total segment length"] = tot.reindex(key_idx).fillna(0).values
        shapes["skeleton length"] = shapes["_skeleton_px"] * sc
        shapes["width"] = shapes["_width_px"] * sc
        for kind in ("branch", "fusion", "crossing"):
            sel = junctions[junctions["kind"] == kind]
            cnt: dict = {}
            for _, j in sel.iterrows():
                for name in str(j["objectName"]).split(", "):
                    cnt[(j["file"], name)] = cnt.get((j["file"], name), 0) + 1
            shapes[f"{kind} points"] = [cnt.get((f, o), 0) for f, o in zip(shapes["file"], shapes["objectName"])]
        shapes = shapes.drop(columns=["_width_px", "_skeleton_px"])
    result = RibbonSet(ribbons, table, shapes, junctions, notes)
    cache[key] = result
    return result


def skeleton_metrics(ctx: AnalysisContext) -> pd.DataFrame:
    """Skeleton length and width per object in ``ctx.objects`` order (default settings)."""
    rs = ribbon_set(ctx, RibbonParams(), SOURCES[1])
    if rs.shapes.empty:
        return pd.DataFrame({"skeleton_length": np.nan, "skeleton_width": np.nan}, index=ctx.objects.index)
    s = rs.shapes.set_index(["file", "objectName"])
    idx = pd.MultiIndex.from_frame(ctx.objects[["file", "objectName"]])
    return pd.DataFrame({"skeleton_length": s["total segment length"].reindex(idx).values,
                         "skeleton_width": s["width"].reindex(idx).values}, index=ctx.objects.index)


# =============================================================================
# Plot helpers
# =============================================================================

def _hist_panel(ax, data: list, bins: int, density: bool, xlabel: str, ctx: AnalysisContext, integer=False):
    data = [(l, np.asarray(v, float)[np.isfinite(v)]) for l, v in data]
    data = [(l, v) for l, v in data if len(v)]
    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    allv = np.concatenate([v for _, v in data])
    if integer:
        lo, hi = int(allv.min()), int(allv.max())
        edges = np.arange(lo - 0.5, hi + 1.5, max(1, int(math.ceil((hi - lo + 1) / max(int(bins), 1)))))
    else:
        edges = np.histogram_bin_edges(allv, bins=int(bins))
    for label, v in data:
        c = ctx.color(label)
        sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
        ax.hist(v, bins=edges, density=density, histtype="stepfilled", alpha=0.3, color=c)
        ax.hist(v, bins=edges, density=density, histtype="step", linewidth=1.5, color=c,
                label=f"{label} (n={len(v)}, {np.mean(v):.3g} ± {sd:.2g})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.legend(fontsize=8)


def _map_figure(ctx: AnalysisContext, file: str, title: str):
    info = ctx.images[file]
    fig = Figure(figsize=(8.2, 7.4), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    img = ctx.image_array(file)
    if img is not None:
        lo, hi = np.nanpercentile(img, [1, 99.5])
        ax.imshow(img, cmap="gray", extent=(0, img.shape[1], img.shape[0], 0), interpolation="nearest",
                  vmin=lo, vmax=hi)
    else:
        ax.set_facecolor("#202020")
    ax.set_xlim(0, info.width)
    ax.set_ylim(info.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return fig, ax


# =============================================================================
# Ribbons: length, width, branching and fusion
# =============================================================================

_PLOTS = ("Segment length and width", "Segment length", "Segment width", "Whole shape skeleton length",
          "Segments per shape", "Connections per segment", "Junction angles", "All")


def _ribbon_param_list():
    return [
        Param("source", "Segments from", "choice", SOURCES[0], choices=SOURCES,
              tooltip="Shapes (outlines and painted masks) are skeletonised and split at junctions and sharp bends.\n"
                      "Centre lines drawn inside a shape replace its skeleton (to correct a wrong split)."),
        Param("kink_angle", "Kink angle (°)", "float", 35.0, 5.0, 120.0,
              tooltip="A segment whose direction turns by more than this within ~2 widths is two fused segments."),
        Param("straight_tolerance", "Straight-through tolerance (°)", "float", 35.0, 0.0, 90.0,
              tooltip="Two branches that continue each other within this angle are one segment passing the junction."),
        Param("spur_length", "Min. side branch (widths)", "float", 1.0, 0.0, 10.0,
              tooltip="Skeleton side branches shorter than this many segment widths are boundary noise, not branches."),
        Param("extend_ends", "Extend free ends", "bool", True,
              tooltip="The skeleton stops half a width before the tip; on: the length runs to the boundary."),
        Param("min_length", "Min. segment length", "float", 0.0, 0.0, 1e6,
              tooltip="Segments shorter than this (nm, or px without a pixel size) are left out of the statistics."),
        Param("line_width", "Centre line width (0 = auto)", "float", 0.0, 0.0, 1e6,
              tooltip="Width assumed for drawn centre lines outside any shape (auto: median shape width)."),
    ]


@register_analysis(
    "skeleton", "Skeleton: length, width, branching",
    "Works on any area ROI (outline or painted mask) of any label. Splits every shape into its separate segments "
    "(skeleton graph, split at junctions and sharp bends) and "
    "measures each segment's length, width and straightness. Counts branch points (a segment ending on the side of "
    "another), fusion points (segments joined end to end at a sharp angle) and crossings, per segment and per shape. "
    "Draw centre lines inside a shape to correct its split.",
    params=_ribbon_param_list() + [
        Param("plots", "Histograms", "choice", _PLOTS[0], choices=_PLOTS),
        Param("bins", "Histogram bins", "int", 20, 2, 200),
        Param("density", "Normalise histograms", "bool", False),
        Param("max_maps", "Maps (images)", "int", 4, 0, 30, tooltip="Images drawn with their split segments and junctions."),
    ],
)
def ribbons_analysis(ctx: AnalysisContext, source=SOURCES[0], kink_angle=35.0, straight_tolerance=35.0,
                     spur_length=1.0, extend_ends=True, min_length=0.0, line_width=0.0, plots=_PLOTS[0], bins=20,
                     density=False, max_maps=4) -> AnalysisResult:
    res = AnalysisResult()
    params = ribbon_params(kink_angle, straight_tolerance, spur_length, extend_ends)
    rs = ribbon_set(ctx, params, source, line_width)
    res.notes.extend(rs.notes)
    t = rs.table
    if t.empty:
        res.notes.append("No segments: the analysis needs area ROIs (polygons, rectangles, ellipses, painted masks) or centre lines.")
        return res
    keep = t["length"] >= float(min_length)
    if (~keep).any():
        res.notes.append(f"{int((~keep).sum())} segment(s) shorter than {min_length:g} {ctx.unit} are left out of the "
                         "statistics (they still count as connections).")
    t = t[keep]
    u = ctx.unit
    # ---- tables
    rows = []
    for label in ctx.labels + [None]:
        sel = t if label is None else t[t["label"] == label]
        sh = rs.shapes if label is None else rs.shapes[rs.shapes["label"] == label]
        jn = rs.junctions if label is None else rs.junctions[rs.junctions["label"] == label]
        if sel.empty:
            continue
        n_sh = len(sh)
        rows.append({
            "label": "All labels" if label is None else label, "shapes": n_sh, "segments": len(sel),
            "segments per shape": len(sel) / n_sh if n_sh else np.nan,
            f"mean length ({u})": sel["length"].mean(), f"std length ({u})": sel["length"].std(ddof=1),
            f"median length ({u})": sel["length"].median(), f"total length ({u})": sel["length"].sum(),
            f"mean width ({u})": sel["width"].mean(), "mean straightness": sel["straightness"].mean(),
            "branch points": int((jn["kind"] == "branch").sum()), "fusion points": int((jn["kind"] == "fusion").sum()),
            "crossings": int((jn["kind"] == "crossing").sum()),
            "unbranched shapes (%)": 100.0 * float((sh["segments"] == 1).mean()) if n_sh else np.nan,
            "branched segments (%)": 100.0 * float(((sel["branches on it"] + sel["branched off another"]) > 0).mean()),
            "fused segments (%)": 100.0 * float((sel["fusion points"] > 0).mean()),
            "branch points per segment": float((jn["kind"] == "branch").sum()) / len(sel),
            "fusion points per segment": float((jn["kind"] == "fusion").sum()) / len(sel),
        })
    res.tables["Summary"] = pd.DataFrame(rows)
    ribbons_out = t.drop(columns=["cx_px", "cy_px"]).rename(columns={
        "length": f"length ({u})", "width": f"width ({u})", "end_to_end": f"end to end ({u})"})
    res.tables["Segments"] = ribbons_out
    res.tables["Shapes"] = rs.shapes.rename(columns={"total segment length": f"total segment length ({u})",
                                                     "skeleton length": f"skeleton length ({u})",
                                                     "width": f"width ({u})"})
    if len(rs.junctions):
        res.tables["Junctions"] = rs.junctions.drop(columns=["_ribbon_ids"])

    # ---- histograms
    labels = [l for l in ctx.labels if (t["label"] == l).any()]
    by = lambda df, col: [(l, df.loc[df["label"] == l, col].to_numpy(float)) for l in labels]  # noqa: E731
    panels = []
    want = _PLOTS[:-1] if plots == "All" else [plots]
    if "Segment length and width" in want:
        want = ["Segment length", "Segment width"] + [w for w in want if w != "Segment length and width"]
    for w in dict.fromkeys(want):
        if w == "Segment length":
            panels.append((w, by(t, "length"), f"Segment length ({u})", False))
        elif w == "Segment width":
            panels.append((w, by(t, "width"), f"Segment width ({u})", False))
        elif w == "Whole shape skeleton length":
            panels.append((w, by(rs.shapes, "total segment length"), f"Total segment length per shape ({u})", False))
        elif w == "Segments per shape":
            panels.append((w, by(rs.shapes, "segments"), "Segments per shape", True))
        elif w == "Connections per segment":
            panels.append((w, by(t, "connections"), "Branch + fusion + crossing points per segment", True))
        elif w == "Junction angles" and len(rs.junctions):
            j = rs.junctions
            panels.append((w, [(k, j.loc[j["kind"] == k, "angle (°)"].to_numpy(float)) for k in
                               ("branch", "fusion", "crossing")], "Smallest angle between the joined segments (°)", False))
    for name, data, xlabel, integer in panels:
        fig, ax = ctx.figure(width=7, height=4.2)
        if name == "Junction angles":
            colors = dict(ctx._colors)
            ctx._colors.update(JUNCTION_COLORS)
            _hist_panel(ax, data, bins, density, xlabel, ctx)
            ctx._colors = colors
        else:
            _hist_panel(ax, data, bins, density, xlabel, ctx, integer)
        ax.set_title(f"{name} (mean ± std)")
        res.figures[name] = fig

    # ---- junction counts
    if len(rs.junctions):
        fig, ax = ctx.figure(width=6, height=3.6)
        kinds = ["branch", "fusion", "crossing"]
        n = [int((rs.junctions["kind"] == k).sum()) for k in kinds]
        ax.bar(kinds, n, color=[JUNCTION_COLORS[k] for k in kinds], edgecolor="black", linewidth=0.5)
        for x, v in enumerate(n):
            ax.annotate(str(v), (x, v), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.set_ylabel("Count")
        ax.set_title(f"Junctions ({len(t)} segments in {len(rs.shapes)} shapes)")
        res.figures["Junction counts"] = fig

    # ---- maps
    from matplotlib.collections import LineCollection
    from matplotlib import colormaps
    files = [f for f in ctx.files if (rs.table["file"] == f).any()][:int(max_maps)]
    cmap = colormaps["tab20"]
    for file in files:
        idx = [k for k in range(len(rs.ribbons)) if rs.table["file"].iat[k] == file]
        fig, ax = _map_figure(ctx, file, f"{file}: {len(idx)} segments (colours: separate segments; "
                                         "red: branch, yellow: fusion, cyan: crossing)")
        segs = [rs.ribbons[k].points for k in idx]
        ax.add_collection(LineCollection(segs, colors=[cmap(i % 20) for i in range(len(segs))], linewidths=2.2))
        jf = rs.junctions[rs.junctions["file"] == file]
        for kind, marker in (("branch", "^"), ("fusion", "o"), ("crossing", "X")):
            s = jf[jf["kind"] == kind]
            if len(s):
                ax.scatter(s["x_px"], s["y_px"], s=46, marker=marker, color=JUNCTION_COLORS[kind],
                           edgecolors="black", linewidths=0.7, zorder=4, label=f"{kind} ({len(s)})")
        if len(jf):
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        res.figures[f"Map {os.path.splitext(file)[0]}"] = fig
    n_files = len([f for f in ctx.files if (rs.table["file"] == f).any()])
    if n_files > len(files):
        res.notes.append(f"Maps drawn for the first {len(files)} images (raise 'Maps (images)' for more).")
    return res


# =============================================================================
# Alignment to the herringbone
# =============================================================================

_REFS = ("Marked herringbone, else detected", "Marked herringbone axes", "Detected from the image",
         "Reference axis", "Image x axis")
_FOLDS = ("Nearest ridge direction (0–90°)", "Nearest of the 3 Au(111) domains (0–30°)")


def _marks(ctx: AnalysisContext, file: str) -> list:
    info = ctx.images[file]
    return list((getattr(info, "marks", None) or {}).get(HERRINGBONE_KIND, []))


def _nearest_mark(marks: list, xy: np.ndarray) -> tuple[float, float]:
    """(direction, distance in px) of the marked axis closest to xy."""
    best = (float("nan"), float("inf"))
    for (x0, y0), (x1, y1) in marks:
        a = np.array([x0, y0], float)
        d = np.array([x1 - x0, y1 - y0], float)
        L2 = float(d @ d)
        t = 0.0 if L2 == 0 else float(np.clip((xy - a) @ d / L2, 0, 1))
        dist = float(np.hypot(*(a + t * d - xy)))
        if dist < best[1]:
            best = (math.degrees(math.atan2(-d[1], d[0])) % 180.0, dist)
    return best


def _mark_points(mark) -> np.ndarray:
    """Points along a marked axis, to find the domain it lies in."""
    (x0, y0), (x1, y1) = mark
    t = np.linspace(0.0, 1.0, 9)[:, None]
    return np.array([x0, y0], float) + t * np.array([x1 - x0, y1 - y0], float)


def _rasterise(ctx: AnalysisContext, file: str, shape: tuple[int, int]) -> np.ndarray:
    """Every ROI of the image (all labels in scope) as a mask, to keep adsorbates out of the herringbone."""
    mask = np.zeros(shape, np.uint8)
    for s in ctx.shapes:
        if s["file"] != file:
            continue
        if s.get("mask") is not None:                   # painted: its pixels
            bits = np.asarray(s["mask"], bool)
            x0, y0 = (int(round(v)) for v in s.get("offset", (0, 0)))
            ys, xs = np.nonzero(bits)
            ys, xs = ys + y0, xs + x0
            ok = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
            mask[ys[ok], xs[ok]] = 1
            continue
        pts = np.round((s["pts"] - 0.5) * 16).astype(np.int32)
        if s["type"] in AREA_TYPES and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1, shift=4)
        elif s["type"] == "LineSegmentROI" and len(pts) >= 2:
            cv2.polylines(mask, [pts], False, 1, thickness=3, shift=4)
    return mask.astype(bool)


def herringbone_for(ctx: AnalysisContext, file: str, period_nm: float = AU111_PERIOD_NM):
    """The detected herringbone field of an image (cached on the context), or None without a scale / image."""
    cache = ctx.__dict__.setdefault("_herringbone", {})
    key = (file, float(period_nm))
    if key not in cache:
        img = ctx.image_array(file)
        px = ctx.images[file].px_scale
        f = None
        if img is not None and px:
            try:
                f = herringbone_field(img, px, period_nm, exclude=_rasterise(ctx, file, img.shape))
            except ValueError:
                f = None
        cache[key] = f
    return cache[key]


def domains_for(ctx: AnalysisContext, file: str, period_nm: float = AU111_PERIOD_NM):
    """The herringbone domains of an image (cached on the context), or None."""
    cache = ctx.__dict__.setdefault("_herringbone_domains", {})
    key = (file, float(period_nm))
    if key not in cache:
        f = herringbone_for(ctx, file, period_nm)
        cache[key] = herringbone_domains(f) if f is not None and f.detected else None
    return cache[key]


@register_analysis(
    "alignment", "Alignment to the herringbone",
    "Angle of every skeleton segment (of any area ROI or centre line) to the Au(111) herringbone ridges next to it. The ridge direction is marked by hand "
    "(Herringbone axis tool, or Analyze > Detect herringbone direction) or measured from the bare surface around "
    "each segment. The image is split into herringbone domains; a marked axis counts for the segments in its domain. "
    "Reports the angle distribution, the share of aligned segments and the order parameter S = <cos 2θ> "
    "(1: all along the ridges, 0: random, -1: all across).",
    params=[
        Param("reference", "Herringbone from", "choice", _REFS[0], choices=_REFS,
              tooltip="Marked, else detected: a segment uses a marked axis of its own herringbone domain, else the "
                      "ridges measured next to it, else its domain's direction.\n"
                      "Marked: herringbone axis annotations only (a mark of the segment's domain, else the nearest).\n"
                      "Detected: ridges measured in the image next to each segment (needs visible bare Au(111)).\n"
                      "Reference axis (a1 of the first lattice) / x axis: one fixed direction for the whole image."),
        Param("fold", "Angle to", "choice", _FOLDS[0], choices=_FOLDS,
              tooltip="The three rotational domains of the reconstruction run 120° apart. 'Nearest of the 3 domains' "
                      "measures each segment against whichever of the three directions is closest."),
        Param("period", "Herringbone period (nm)", "float", AU111_PERIOD_NM, 2.0, 30.0,
              tooltip="Spacing of neighbouring ridge pairs; 6.3 nm on Au(111)."),
        Param("radius", "Search radius (nm)", "float", 8.0, 1.0, 200.0,
              tooltip="Detected: bare surface within this distance of the segment gives its local ridge direction."),
        Param("aligned_within", "Aligned within (°)", "float", 15.0, 1.0, 45.0),
        Param("weighted", "Weight by segment length", "bool", False,
              tooltip="Histogram and statistics weight each segment by its length (long segments count more)."),
        Param("bins", "Histogram bins", "int", 18, 2, 90),
        Param("max_maps", "Maps (images)", "int", 4, 0, 30),
    ] + _ribbon_param_list(),
)
def alignment_analysis(ctx: AnalysisContext, reference=_REFS[0], fold=_FOLDS[0], period=AU111_PERIOD_NM, radius=8.0,
                       aligned_within=15.0, weighted=False, bins=18, max_maps=4, source=SOURCES[0], kink_angle=35.0,
                       straight_tolerance=35.0, spur_length=1.0, extend_ends=True, min_length=0.0,
                       line_width=0.0) -> AnalysisResult:
    res = AnalysisResult()
    params = ribbon_params(kink_angle, straight_tolerance, spur_length, extend_ends)
    rs = ribbon_set(ctx, params, source, line_width)
    res.notes.extend(rs.notes)
    t = rs.table
    if t.empty:
        res.notes.append("No segments: the analysis needs area ROIs (polygons, rectangles, ellipses, painted masks) or centre lines.")
        return res
    nfold = 3 if fold == _FOLDS[1] else 1
    period_deg = 90.0 / nfold
    ref_angle = np.full(len(t), np.nan)
    ref_from = np.array([""] * len(t), dtype=object)
    strength = np.full(len(t), np.nan)
    no_ref: dict[str, str] = {}
    for file in ctx.files:
        idx = np.flatnonzero(t["file"].to_numpy() == file)
        if not len(idx):
            continue
        marks = _marks(ctx, file)
        field_, dom = None, None
        if reference in _REFS[:3] and ctx.images[file].px_scale:
            field_ = herringbone_for(ctx, file, period)
            if field_ is not None and field_.detected:
                dom = domains_for(ctx, file, period)
            else:
                field_ = None
        if reference == _REFS[2] or (reference == _REFS[0] and not marks):
            if not ctx.images[file].px_scale:
                no_ref[file] = "no pixel size"
            elif field_ is None:
                no_ref[file] = "no herringbone found in the image"
        if field_ is not None and reference != _REFS[1]:
            res.notes.append(f"{file}: ridges detected along " + ", ".join(
                f"{a:.0f}° ({100 * s:.0f}%)" for a, s in field_.peaks)
                + (f" in {len(dom)} domain(s)" if dom is not None and len(dom) else "")
                + (f", period {field_.period_nm:.1f} nm" if field_.period_nm else ""))
        if reference == _REFS[1] and not marks:
            no_ref[file] = "no herringbone axis marked"
        # a marked axis speaks for the domain it lies in: a ribbon uses a mark of its own domain, else
        # the ridges measured next to it, else its domain's direction; the nearest mark only without a domain
        mark_dom = [dom.domain_of(_mark_points(m)) if dom is not None else 0 for m in marks]
        rad = radius / ctx.scale(file) if ctx.unit == "nm" else radius / (ctx.images[file].px_scale or 1.0)
        for k in idx:
            r = rs.ribbons[k]
            centre = r.points.mean(axis=0)
            d = dom.domain_of(r.points) if dom is not None else 0
            if reference in (_REFS[0], _REFS[1]) and marks:
                own = [m for m, md in zip(marks, mark_dom) if d and md == d]
                if own or reference == _REFS[1] or field_ is None:
                    ref_angle[k], _ = _nearest_mark(own or marks, centre)
                    ref_from[k], strength[k] = ("marked" if own or dom is None else "marked (other domain)"), 1.0
                    continue
            if field_ is not None:
                a, st = field_.direction_near(r.points, rad + r.width / 2.0)
                if math.isfinite(a):
                    ref_angle[k], strength[k], ref_from[k] = a, st, "detected"
                elif d and dom is not None:
                    ref_angle[k], strength[k], ref_from[k] = dom.angles[d - 1], 0.0, "detected (domain)"
            elif reference == _REFS[3]:
                ra = ctx.images[file].ref_angle
                if ra is None:
                    no_ref[file] = "no reference axis"
                else:
                    ref_angle[k], ref_from[k], strength[k] = ra % 180.0, "reference axis", 1.0
            elif reference == _REFS[4]:
                ref_angle[k], ref_from[k], strength[k] = 0.0, "x axis", 1.0
    for file, why in no_ref.items():
        res.notes.append(f"{file}: {why}; its segments have no alignment angle.")
    rel = axial_difference(t["angle (°)"].to_numpy(float), ref_angle, nfold)
    table = t[["file", "objectName", "label", "segment", "length", "width", "angle (°)"]].copy()
    table["herringbone (°)"] = ref_angle
    table["from"] = ref_from
    table["strength"] = strength
    table["angle to herringbone (°)"] = rel
    keep = np.isfinite(rel) & (t["length"].to_numpy(float) >= float(min_length))
    missing = int((np.isfinite(t["angle (°)"].to_numpy(float)) & ~np.isfinite(rel)).sum())
    if missing and not no_ref:
        res.notes.append(f"{missing} segment(s) have no bare surface with a clear ridge pattern within "
                         f"{radius:g} nm: no alignment angle (mark the herringbone by hand there).")
    u = ctx.unit
    res.tables["Segments"] = table.rename(columns={"length": f"length ({u})", "width": f"width ({u})",
                                                  "angle (°)": "segment angle (°)"})
    if not keep.any():
        res.notes.append("No segment has an alignment angle.")
        return res
    lengths = t["length"].to_numpy(float)
    rows = []
    for label in ctx.labels + [None]:
        sel = keep & ((t["label"] == label).to_numpy() if label is not None else True)
        if not sel.any():
            continue
        a = rel[sel]
        w = lengths[sel] if weighted else np.ones(sel.sum())
        w = w / w.sum()
        S = float((w * np.cos(np.radians(2 * a * nfold))).sum())
        rows.append({"label": "All labels" if label is None else label, "segments": int(sel.sum()),
                     "mean angle (°)": float((w * a).sum()), "median angle (°)": float(np.median(a)),
                     f"aligned ≤{aligned_within:g}° (%)": 100.0 * float(w[a <= aligned_within].sum()),
                     f"across ≥{period_deg - aligned_within:g}° (%)": 100.0 * float(w[a >= period_deg - aligned_within].sum()),
                     "order parameter S": S,
                     "marked / detected": f"{sum(str(v).startswith('marked') for v in ref_from[sel])} / "
                                          f"{sum(str(v).startswith('detected') for v in ref_from[sel])}"})
    res.tables["Summary"] = pd.DataFrame(rows)

    # ---- histogram
    labels = [l for l in ctx.labels if (keep & (t["label"] == l).to_numpy()).any()]
    edges = np.linspace(0, period_deg, int(bins) + 1)
    fig, ax = ctx.figure(width=7, height=4)
    data = [rel[keep & (t["label"] == l).to_numpy()] for l in labels]
    wts = [lengths[keep & (t["label"] == l).to_numpy()] for l in labels] if weighted else None
    ax.hist(data, bins=edges, weights=wts, stacked=True, color=[ctx.color(l) for l in labels],
            label=[str(l) for l in labels], edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, period_deg)
    ax.set_xlabel("Angle between segment and herringbone ridges (°)" +
                  (" — nearest of 3 domains" if nfold == 3 else ""))
    ax.set_ylabel(f"Total length ({u})" if weighted else "Segments")
    ax.axvspan(0, aligned_within, color="#2ca02c", alpha=0.08)
    ax.legend(fontsize=8)
    ax.set_title("Alignment to the herringbone (0° = along the ridges)", fontsize=10)
    res.figures["Angle to herringbone"] = fig

    # ---- polar half rose of the absolute directions with the reference(s)
    fig = Figure(figsize=(5.2, 3.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ang = t["angle (°)"].to_numpy(float)[keep]
    e2 = np.linspace(0, 180, 37)
    cnt, _ = np.histogram(ang, bins=e2, weights=lengths[keep] if weighted else None)
    ax.bar(np.radians(e2[:-1]), cnt, width=np.radians(5), align="edge", color="#7f7f7f", alpha=0.8)
    refs = ref_angle[keep]
    rc, _ = np.histogram(refs[np.isfinite(refs)], bins=e2)
    if rc.max() > 0 and cnt.max() > 0:
        ax.bar(np.radians(e2[:-1]), rc / rc.max() * cnt.max(), width=np.radians(5), align="edge",
               color="none", edgecolor="#E8412C", linewidth=1.2)
    ax.set_thetamin(0)  # type: ignore[attr-defined]
    ax.set_thetamax(180)  # type: ignore[attr-defined]
    ax.set_yticks([])
    ax.set_title("Segment directions (grey) and herringbone directions (red outline), 0° = x axis", fontsize=8)
    res.figures["Directions"] = fig

    # ---- maps
    from matplotlib import colormaps
    from matplotlib.collections import LineCollection
    cmap = colormaps["viridis_r"]
    files = [f for f in ctx.files if (keep & (t["file"] == f).to_numpy()).any()][:int(max_maps)]
    for file in files:
        idx = np.flatnonzero((t["file"] == file).to_numpy())
        fig, ax = _map_figure(ctx, file, f"{file}: segments coloured by angle to the herringbone "
                                         "(yellow = along, purple = across; grey = none)")
        info = ctx.images[file]
        fld = herringbone_for(ctx, file, period) if reference in (_REFS[0], _REFS[2]) else None
        if fld is not None and fld.detected:
            dm = domains_for(ctx, file, period)
            if dm is not None and len(dm) > 1:                   # domain borders
                ax.contour(dm.labels, levels=np.arange(1.5, len(dm) + 0.5), colors="#FF6E40", linewidths=0.6,
                           alpha=0.5)
            st = max(8, int(min(info.width, info.height) / 28))
            yy, xx = np.mgrid[st // 2:fld.angle.shape[0]:st, st // 2:fld.angle.shape[1]:st]
            ok = fld.valid[yy, xx] & (fld.coherence[yy, xx] > 0.25)
            a = np.radians(fld.angle[yy, xx][ok])
            ax.quiver(xx[ok], yy[ok], np.cos(a), -np.sin(a), angles="xy", color="#FF6E40", pivot="middle", headwidth=0,
                      headlength=0, headaxislength=0, scale=1.6 / st, scale_units="xy", width=0.0025, alpha=0.8)
        for (x0, y0), (x1, y1) in _marks(ctx, file):
            ax.plot([x0, x1], [y0, y1], color="#FF6E40", lw=2.5)
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-", color="#FF6E40", lw=2.5))
        colors = [cmap(rel[k] / period_deg) if np.isfinite(rel[k]) else (0.6, 0.6, 0.6, 1) for k in idx]
        ax.add_collection(LineCollection([rs.ribbons[k].points for k in idx], colors=colors, linewidths=2.4))
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        fig.colorbar(ScalarMappable(Normalize(0, period_deg), cmap), ax=ax, shrink=0.6,
                     label="Angle to herringbone (°)")
        res.figures[f"Map {os.path.splitext(file)[0]}"] = fig
    return res
