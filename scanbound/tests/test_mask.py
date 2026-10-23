"""Mask features: exact pixels (holes included) from the paint tool and the adapt pipeline, saved with the
project, drawn on the canvas and measured exactly by the analyses."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DATA = os.path.join(os.path.dirname(__file__), "data")


def _ring():
    b = np.zeros((6, 7), bool)
    b[1:5, 1:6] = True
    b[2:4, 3] = False                                                     # a 1 x 2 hole
    return b


def test_mask_geometry_is_exact_and_round_trips():
    from scanbound.annotations import Mask, geometry_from_dict
    m = Mask(10, 20, _ring())
    assert (m.x0, m.y0, m.bits.shape) == (11, 21, (4, 5))                 # cropped to its pixels
    assert m.area() == 18 and m.length() == 2 * (5 + 4) + 2 * (1 + 2)     # outer + hole edges
    assert m.area((2.0, 0.5)) == 18.0
    outer, holes = m.contours
    assert outer[0].tolist() == [[11, 21], [11, 25], [16, 25], [16, 21]] and len(holes) == 1
    assert m.contains(11.5, 21.5) and not m.contains(13.5, 22.5)           # in the hole
    assert geometry_from_dict(m.to_dict()) == m
    assert m.translated(2.4, -1).bbox() == (13.0, 20.0, 18.0, 24.0)
    assert m.transformed(np.diag([2.0, 1.0, 1.0])).area() == 36
    assert m.mask((30, 30)).sum() == 18 and m.clipped((22, 13)).area() == 2
    assert m.combined(Mask(11, 21, np.ones((2, 2), bool)), subtract=True).area() == 14


def test_adapt_mask_rois_keep_every_pixel():
    from scanbound.proposals import object_geometry
    from scanbound.roi import RoiSettings, extract_objects
    m = np.zeros((40, 50), bool)
    m[5:15, 5:20] = True
    m[8:11, 9:12] = False
    m[25:30, 30:45] = True
    cache: dict = {}
    for _ in range(2):                                                    # the second call uses the cache
        objs = extract_objects(m, RoiSettings(roi_type="MaskROI"), cache)
    geoms = [object_geometry(o)[1] for o in objs]
    assert sorted(g.area() for g in geoms) == [75, 141]
    total = np.zeros_like(m)
    for g in geoms:
        total |= g.mask(m.shape)
    assert np.array_equal(total, m)


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def session(qapp, tmp_path):
    from scanbound.app.session import Session
    shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / "scan.sxm")
    s = Session()
    s.open_folder(str(tmp_path))
    yield s
    s.project.dirty = False


def test_paint_tool_paints_erases_and_undoes(session):
    from scanbound.annotations import Mask
    from scanbound.app.tools import TOOLS
    t = TOOLS["paint"](session, lambda: 1.0)
    t.radius = 3.0
    t.press(10, 10); t.move(30, 10); t.release(30, 10)
    (a,) = session.annotations()
    assert a.kind == "feature" and isinstance(a.geometry, Mask) and a.geometry.area() > 100
    painted = a.geometry.area()
    t.press(20, 10, "right"); t.release(20, 10, "right")                  # erase a dab from the selected mask
    assert session.annotations()[0].geometry.area() < painted
    session.select([])
    t.press(80, 30); t.release(80, 30)                                    # elsewhere: a new mask
    assert len(session.annotations()) == 2
    t.press(80, 30, "left", frozenset({"ctrl"})); t.release(80, 30)        # erase all of it: it is removed
    assert len(session.annotations()) == 1
    session.store.undo(); session.store.undo(); session.store.undo()
    assert [x.geometry.area() for x in session.annotations()] == [painted]
    t.key("]")
    assert t.radius > 3.0


def test_mask_project_canvas_and_analysis(session, tmp_path):
    from scanbound.analysis.adapter import analysis_context
    from scanbound.annotations import Annotation, Mask
    from scanbound.app.mainwindow import MainWindow
    from scanbound.project import Project
    b = np.zeros((12, 20), bool)
    b[2:10, 3:18] = True
    b[5:7, 8:12] = False
    m = Mask(30, 10, b)
    session.store.add(Annotation(session.current_id, "feature", m, label="blob"), "add")
    session.save()
    p = Project.load(session.project.path)
    (back,) = [a for a in p.annotations if a.kind == "feature"]
    assert back.geometry == m
    ctx = analysis_context(session.project)
    row = ctx.objects.iloc[0]
    img = session.image()
    if img.isotropic:
        assert row["area_px"] == m.area() and row["type"] == "MaskROI"
    assert ctx.shapes[0]["mask"] is not None
    w = MainWindow(session)
    w.show()
    w.canvas.set_tool("paint")
    w.canvas.viewport().grab()                                            # paints the mask without errors
    session.project.dirty = False
    w.close()


def test_mask_rois_can_be_grown():
    from scanbound.proposals import object_geometry
    from scanbound.roi import RoiSettings, extract_objects
    m = np.zeros((40, 50), bool)
    m[5:15, 5:20] = True
    m[0:3, 46:50] = True                                                  # at the image corner
    cache: dict = {}
    for _ in range(2):
        objs = extract_objects(m, RoiSettings(roi_type="MaskROI", mask_grow=2.0), cache)
    big, corner = sorted((object_geometry(o)[1] for o in objs), key=lambda g: -g.area())
    assert big.bbox() == (3.0, 3.0, 22.0, 17.0) and big.area() > 150
    assert corner.bbox()[2:] == (50.0, 5.0) and corner.bbox()[:2] == (44.0, 0.0)   # clipped to the image


def test_adapt_single_label_is_never_a_stale_label(qapp):
    from scanbound.app.widgets.label_assignment_panel import LabelAssignmentPanel
    p = LabelAssignmentPanel(["9AP"], {}, None, None)
    assert p.config()["single_label"] is None                            # not the first label of the list
    p = LabelAssignmentPanel([], {}, None, "molecule")
    p.set_config({"mode": "single", "single_label": "9AP"})              # saved in an earlier project
    assert p.config()["single_label"] == "molecule"
