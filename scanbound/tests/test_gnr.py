"""Shape skeletons (outlines and painted masks): skeleton graph and segment split, branch / fusion / crossing points, herringbone
direction and alignment, and the GUI actions that propose centre lines and herringbone axes."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from scanbound.analysis import analysis_context, run_analyses
from scanbound.annotations import Annotation, Polygon, Polyline, Segment
from scanbound.herringbone import (axial_difference, herringbone_axes, herringbone_domains,
                                   herringbone_field)
from scanbound.project import Project
from scanbound.ribbon_graph import (RibbonParams, mask_ribbons, polygon_ribbons, ribbon_connections, ribbons_from_polyline,
                                    skeleton_graph)


def outline(lines, width, size=400):
    """Polygon outline of the union of thick polylines (round caps)."""
    m = np.zeros((size, size), np.uint8)
    for line in lines:
        cv2.polylines(m, [np.int32(line)], False, 1, thickness=width)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0, :].astype(float) + 0.5


def split(lines, width=12):
    _g, ribbons = polygon_ribbons(outline(lines, width))
    return ribbons, ribbon_connections(ribbons)


def kinds(junctions):
    return sorted(j.kind for j in junctions)


# ---- skeleton graph and split ----------------------------------------------------------------

def test_straight_ribbon_length_includes_the_tips():
    ribbons, junctions = split([[(50, 200), (350, 200)]])
    assert len(ribbons) == 1 and not junctions
    assert ribbons[0].contour_length == pytest.approx(312, rel=0.01)       # 300 + two round caps of 6
    assert ribbons[0].width == pytest.approx(13, abs=1.5)
    assert ribbons[0].axis_angle == pytest.approx(0, abs=1) or ribbons[0].axis_angle == pytest.approx(180, abs=1)


def test_oblique_length_is_not_inflated_by_the_pixel_staircase():
    ribbons, _ = split([[(60, 60), (300, 200)]])
    true = math.hypot(240, 140) + 12
    assert ribbons[0].contour_length == pytest.approx(true, rel=0.02)


def test_t_junction_is_one_branch_on_a_through_ribbon():
    ribbons, junctions = split([[(50, 200), (350, 200)], [(200, 200), (200, 50)]])
    assert len(ribbons) == 2 and kinds(junctions) == ["branch"]
    j = junctions[0]
    through = ribbons[j.sides[0]]
    assert through.contour_length == pytest.approx(312, rel=0.02)
    assert j.angle == pytest.approx(90, abs=5)


def test_sharp_bend_is_a_fusion_of_two_ribbons():
    ribbons, junctions = split([[(50, 300), (200, 100), (350, 300)]])
    assert len(ribbons) == 2 and kinds(junctions) == ["fusion"]
    assert len(junctions[0].ends) == 2


def test_gentle_curve_stays_one_ribbon():
    pts = [(50 + i * 3, 200 + 60 * np.sin(i / 30)) for i in range(100)]
    ribbons, junctions = split([pts])
    assert len(ribbons) == 1 and not junctions


def test_crossing_and_three_way_fusion():
    ribbons, junctions = split([[(50, 200), (350, 200)], [(120, 80), (280, 320)]])
    assert len(ribbons) == 2 and kinds(junctions) == ["crossing"]
    ribbons, junctions = split([[(200, 200), (200, 50)], [(200, 200), (60, 300)], [(200, 200), (340, 300)]])
    assert len(ribbons) == 3 and kinds(junctions) == ["fusion"] and len(junctions[0].ends) == 3


def test_kink_threshold_is_a_setting():
    poly = outline([[(50, 300), (200, 200), (350, 300)]], 12)      # turns by ~67°
    assert len(polygon_ribbons(poly, RibbonParams(kink_angle=35))[1]) == 2
    assert len(polygon_ribbons(poly, RibbonParams(kink_angle=80, straight_tolerance=80))[1]) == 1


def test_bumps_on_the_outline_are_not_branches():
    m = np.zeros((300, 400), np.uint8)
    cv2.polylines(m, [np.int32([(50, 150), (350, 150)])], False, 1, thickness=14)
    for x in (120, 200, 280):
        cv2.circle(m, (x, 158), 5, 1, -1)                          # small blobs on one side
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    g = skeleton_graph(max(cs, key=cv2.contourArea)[:, 0, :] + 0.5)
    assert not g.junctions()


def test_drawn_lines_join_as_branches_and_fusions():
    a = ribbons_from_polyline(np.array([(0, 0), (100, 0)], float), 6, source="a")
    b = ribbons_from_polyline(np.array([(50, 1), (50, 60)], float), 6, source="b")       # ends on a's side
    c = ribbons_from_polyline(np.array([(100, 1), (140, 40)], float), 6, source="c")     # ends on a's end
    assert kinds(ribbon_connections(a + b + c)) == ["branch", "fusion"]


def test_painted_mask_splits_like_its_outline():
    m = np.zeros((60, 200), np.uint8)
    cv2.line(m, (10, 30), (190, 30), 1, 5)
    cv2.line(m, (100, 30), (100, 55), 1, 5)
    _g, segs = mask_ribbons(m.astype(bool), (50, 20))
    assert sorted(round(s.contour_length) for s in segs) == [27, 186]
    assert kinds(ribbon_connections(segs)) == ["branch"]
    y = np.concatenate([s.points[:, 1] for s in segs])
    assert y.min() > 49 and y.max() < 79                     # scene = mask pixel + offset (pixel centres)


def test_ring_mask_keeps_its_hole():
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 50), 30, 1, 6)
    _g, segs = mask_ribbons(m.astype(bool))
    assert sum(s.contour_length for s in segs) == pytest.approx(2 * math.pi * 30, rel=0.05)


# ---- herringbone -----------------------------------------------------------------------------

def stripes(angle_deg, px_nm=0.2, size=320, period_nm=6.3, seed=0):
    """Ridges along a screen angle (counter-clockwise from +x, y down) plus noise."""
    y, x = np.mgrid[0:size, 0:size].astype(float)
    t = math.radians(angle_deg)
    across = (x * math.sin(t) + y * math.cos(t)) * px_nm          # distance across the ridges
    rng = np.random.default_rng(seed)
    return 0.02 * np.cos(2 * np.pi * across / period_nm) + 0.004 * rng.standard_normal((size, size))


@pytest.mark.parametrize("angle", [0, 30, 75, 120, 160])
def test_herringbone_direction_is_found(angle):
    img = stripes(angle)
    img[150:170, 40:280] += 0.2                                       # a ribbon on top, masked out
    mask = np.zeros(img.shape, bool)
    mask[150:170, 40:280] = True
    f = herringbone_field(img, 0.2, exclude=mask)
    assert f.detected
    assert axial_difference(f.peaks[0][0], angle) < 3
    if angle:                         # ridges along the fast scan look partly like line offsets
        assert f.period_nm == pytest.approx(6.3, rel=0.25)
    (x0, y0), (x1, y1), _ = herringbone_axes(f)[0]
    assert axial_difference(math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180, angle) < 4


def three_domains(size=480, px_nm=0.2):
    """Three bands of ridges along 30°, 90° and 150° (left to right)."""
    img = np.zeros((size, size))
    for k, angle in enumerate((30, 90, 150)):
        img[:, k * size // 3:(k + 1) * size // 3] = stripes(angle, px_nm, size, seed=k)[:, k * size // 3:(k + 1) * size // 3]
    return img


def test_domains_cover_the_whole_image():
    f = herringbone_field(three_domains(), 0.2)
    dom = herringbone_domains(f)
    assert len(dom) >= 3 and (dom.labels > 0).all()
    for k, angle in enumerate((30, 90, 150)):                  # each band is a domain of its own direction
        d = dom.domain_of(np.array([[80 + 160 * k, 240]]))
        assert d and axial_difference(dom.angles[d - 1], angle) < 5
    axes = herringbone_axes(f, domains=dom)
    for k, angle in enumerate((30, 90, 150)):                  # every band gets axes along its ridges
        mine = [a for a in axes if 160 * k < (a[0][0] + a[1][0]) / 2 < 160 * (k + 1)]
        assert len(mine) >= 2
        for (x0, y0), (x1, y1), _ in mine:
            assert axial_difference(math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180, angle) < 8


def test_small_image_is_not_searched():
    f = herringbone_field(np.random.default_rng(0).standard_normal((64, 64)), 0.01)   # 0.6 nm wide
    assert not f.detected


def test_no_herringbone_on_noise():
    rng = np.random.default_rng(1)
    f = herringbone_field(rng.standard_normal((256, 256)), 0.2)
    assert not f.detected


def test_axial_difference_folds():
    assert axial_difference(170, 10) == pytest.approx(20)
    assert axial_difference(100, 10) == pytest.approx(90)
    assert axial_difference(70, 10, fold=3) == pytest.approx(0)      # 60° apart: another Au(111) domain


# ---- analyses on a project -------------------------------------------------------------------

@pytest.fixture()
def gnr_project(tmp_path):
    img = stripes(30, px_nm=0.2, size=320)
    arr = ((img - img.min()) / np.ptp(img) * 200).astype(np.uint8)
    PILImage.fromarray(arr).save(tmp_path / "gnr.png")
    p = Project.open_folder(str(tmp_path))
    iid = p.entry(p.image_ids()[0]).id
    p.set_calibration(iid, 0.2e-9, 0.2e-9)
    # a T-shaped fused pair and a single ribbon along 30°
    t = outline([[(40, 60), (280, 60)], [(160, 60), (160, 200)]], 8, size=320)
    t = cv2.approxPolyDP(np.float32(t).reshape(-1, 1, 2), 0.5, True)[:, 0, :].astype(float)
    p.annotations.add(Annotation(iid, "feature", Polygon(t), label="GNR"))
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    single = outline([[(100, 290), (100 + 150 * c, 290 - 150 * s)]], 8, size=320)
    single = cv2.approxPolyDP(np.float32(single).reshape(-1, 1, 2), 0.5, True)[:, 0, :].astype(float)
    p.annotations.add(Annotation(iid, "feature", Polygon(single), label="GNR"))
    return p, iid


def test_ribbon_analysis_tables_and_bins(gnr_project):
    p, iid = gnr_project
    ctx = analysis_context(p, [iid])
    res = run_analyses(ctx, ["skeleton"], {"skeleton": {"bins": 7, "plots": "All"}})["skeleton"]
    assert not [n for n in res.notes if "failed" in n]
    rib = res.tables["Segments"]
    assert len(rib) == 3
    assert rib["branches on it"].sum() == 1 and rib["branched off another"].sum() == 1
    summary = res.tables["Summary"].set_index("label")
    assert summary.loc["GNR", "branch points"] == 1 and summary.loc["GNR", "shapes"] == 2
    longest = rib["length (nm)"].max()
    assert longest == pytest.approx(248 * 0.2, rel=0.03)              # 240 px + caps, in nm
    ax = res.figures["Segment length"].axes[0]
    assert len(ax.patches[0].get_xy()) >= 2 * 7                        # the bins setting is used (step outline)
    assert "Map gnr" in res.figures


def test_skeleton_analysis_uses_painted_masks(gnr_project):
    p, iid = gnr_project
    ctx = analysis_context(p, [iid])
    for sh in ctx.shapes:                                      # the same shapes, painted instead of outlined
        x0, y0 = np.floor(sh["pts"].min(axis=0)).astype(int) - 1
        x1, y1 = np.ceil(sh["pts"].max(axis=0)).astype(int) + 1
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round((sh["pts"] - [x0, y0] - 0.5) * 16).astype(np.int32)], 1, shift=4)
        sh["mask"], sh["offset"] = m.astype(bool), (int(x0), int(y0))
    res = run_analyses(ctx, ["skeleton"])["skeleton"]
    rib = res.tables["Segments"]
    assert len(rib) == 3 and rib["branches on it"].sum() == 1
    assert rib["length (nm)"].max() == pytest.approx(248 * 0.2, rel=0.04)


def test_size_analysis_offers_skeleton_length(gnr_project):
    p, iid = gnr_project
    ctx = analysis_context(p, [iid])
    res = run_analyses(ctx, ["size"], {"size": {"metric": "skeleton_length", "bins": 5}})["size"]
    stats = res.tables["Statistics"]
    row = stats[(stats["metric"] == "Skeleton length") & (stats["label"] == "GNR")].iloc[0]
    assert row["n"] == 2 and row["max"] == pytest.approx((248 + 136) * 0.2, rel=0.05)
    assert "Skeleton length" in res.figures
    objects = res.tables["Objects"]                  # the per-ROI measurement table is part of it
    assert len(objects) >= 2 and any(c.startswith("skeleton length") for c in objects.columns)


def test_drawn_centre_lines_replace_the_skeleton(gnr_project):
    p, iid = gnr_project
    # correct the T: say it is one bent ribbon plus a short separate piece
    p.annotations.add(Annotation(iid, "centre_line", Polyline([(44, 60), (160, 60), (160, 196)]), label="GNR"))
    ctx = analysis_context(p, [iid])
    res = run_analyses(ctx, ["skeleton"])["skeleton"]
    shapes = res.tables["Shapes"]
    t_shape = shapes[shapes["segments from"] == "centre lines"]
    assert len(t_shape) == 1 and int(t_shape["segments"].iloc[0]) == 2   # split at the 90° bend
    assert int(t_shape["fusion points"].iloc[0]) == 1


def test_alignment_marked_and_detected(gnr_project):
    p, iid = gnr_project
    ctx = analysis_context(p, [iid])
    res = run_analyses(ctx, ["alignment"], {"alignment": {"reference": "Detected from the image"}})["alignment"]
    rib = res.tables["Segments"]
    single = rib.loc[rib["segment angle (°)"].sub(30).abs() < 5]
    assert len(single) == 1
    assert single["angle to herringbone (°)"].iloc[0] < 5
    assert single["from"].iloc[0] == "detected"
    # a marked axis wins over detection
    p.annotations.add(Annotation(iid, "herringbone_axis", Segment([(20, 20), (120, 20)])))
    ctx = analysis_context(p, [iid])
    res = run_analyses(ctx, ["alignment"])["alignment"]
    rib = res.tables["Segments"]
    assert set(rib["from"]) == {"marked"}
    horiz = rib.loc[(rib["segment angle (°)"] % 180).sub(0).abs().lt(3) | (rib["segment angle (°)"] > 177)]
    assert (horiz["angle to herringbone (°)"] < 3).all()
    s = res.tables["Summary"]
    assert "order parameter S" in s.columns


def test_a_mark_counts_only_in_its_own_domain(tmp_path):
    img = three_domains()
    arr = ((img - img.min()) / np.ptp(img) * 200).astype(np.uint8)
    PILImage.fromarray(arr).save(tmp_path / "domains.png")
    p = Project.open_folder(str(tmp_path))
    iid = p.entry(p.image_ids()[0]).id
    p.set_calibration(iid, 0.2e-9, 0.2e-9)
    for x in (80, 400):                                          # a horizontal ribbon in the left and right bands
        poly = outline([[(x - 50, 240), (x + 50, 240)]], 8, size=480)
        p.annotations.add(Annotation(iid, "feature", Polygon(poly), label="GNR"))
    p.annotations.add(Annotation(iid, "herringbone_axis", Segment([(60, 100), (100, 77)])))   # 30°, left band only
    res = run_analyses(analysis_context(p, [iid]), ["alignment"])["alignment"]
    rib = res.tables["Segments"].sort_values("objectName")
    left, right = (rib.loc[rib["file"].notna()].iloc[i] for i in (0, 1))
    if left["from"] != "marked":
        left, right = right, left
    assert left["from"] == "marked" and left["herringbone (°)"] == pytest.approx(30, abs=1)
    assert right["from"] == "detected" and axial_difference(right["herringbone (°)"], 150) < 6


def test_herringbone_axis_survives_the_old_csv(gnr_project, tmp_path):
    from scanbound.annotations.legacy import from_frame, to_frame
    p, iid = gnr_project
    p.annotations.add(Annotation(iid, "herringbone_axis", Segment([(20, 20), (120, 40)])))
    df = to_frame(list(p.annotations))
    back, problems = from_frame(df, lambda name: (name, None))
    assert not problems
    assert [a.kind for a in back].count("herringbone_axis") == 1


# ---- GUI -------------------------------------------------------------------------------------

def test_gui_tool_split_and_detect(gnr_project, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.app.tools import TOOLS
    p, iid = gnr_project
    p.save()
    s = Session()
    s.open_project(p.path)
    w = MainWindow(s)
    app.processEvents()
    tool = TOOLS["herringbone"](s, lambda: 1.0)
    tool.press(10, 10); tool.release(60, 10)
    tool.press(10, 30); tool.release(60, 40)                          # several allowed (one per domain)
    assert len(s.store.for_image(s.current_id, kinds=("herringbone_axis",))) == 2
    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", staticmethod(lambda *a, **k: (a[3], True)))
    s.select([])
    w.split_ribbons()
    lines = s.store.for_image(s.current_id, kinds=("centre_line",))
    assert len(lines) == 3 and all(a.status == "accepted" and a.parent for a in lines)   # review is off
    w.split_ribbons()                                                  # re-running leaves them alone
    assert len(s.store.for_image(s.current_id, kinds=("centre_line",))) == 3
    w.detect_herringbone()
    proposed = [a for a in s.store.for_image(s.current_id, kinds=("herringbone_axis",))
                if a.status == "accepted" and a.origin.method == "detector"]
    assert proposed
    w.detect_herringbone()                                             # again: replaces its untouched axes
    assert len([a for a in s.store.for_image(s.current_id, kinds=("herringbone_axis",))
                if a.origin.method == "detector"]) == len(proposed)
    (x0, y0), (x1, y1) = proposed[0].geometry.points
    assert axial_difference(math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180, 30) < 5
    s.project.dirty = False
    w.close()


def test_wide_saved_layout_is_stacked(monkeypatch):
    """A saved layout with the panels side by side can need more than the screen: they become tabs."""
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    from scanbound.app.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = MainWindow()
    right = list(w.docks.values())[:6]
    for d in right:
        d.setFloating(False)
        w.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, d)
        d.show()
    for a, b in zip(right, right[1:]):
        w.splitDockWidget(a, b, Qt.Orientation.Horizontal)
    assert w.minimumSizeHint().width() > 1900
    w._fit_panels(1900, 1050)
    assert w.minimumSizeHint().width() <= 1900
    w.close()
