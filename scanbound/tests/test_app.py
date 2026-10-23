"""GUI layer: tools edit the store through the session; the canvas and main window build and run
offscreen on real data."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def session(qapp, tmp_path):
    from scanbound.app.session import Session
    shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / "scan.sxm")
    shutil.copy(os.path.join(DATA, "bias_spec.dat"), tmp_path / "spec.dat")
    s = Session()
    s.open_folder(str(tmp_path))
    return s


def tools(session):
    from scanbound.app.tools import TOOLS
    return {k: cls(session, lambda: 1.0) for k, cls in TOOLS.items()}


def test_session_opens_folder_and_lists_images(session):
    assert session.image_ids() == ["scan.sxm::"]
    assert session.current_id.startswith("scan.sxm::")
    assert session.image().shape == (50, 144)
    assert "spec.dat" in session.project.spectra_files()


def test_drawing_tools_create_annotations_of_their_kind(session):
    t = tools(session)
    session.set_current_label("molecule")
    t["rect"].press(10, 10); t["rect"].move(30, 20); t["rect"].release(30, 20)
    t["rect"].press(5, 5); t["rect"].release(5.1, 5.1)                       # a click is not a box
    t["point"].press(40, 40)
    for p in [(60, 5), (80, 5), (80, 25)]:
        t["polygon"].press(*p)
    t["polygon"].key("enter")
    for p in [(90, 10), (100, 20), (110, 15)]:
        t["centre_line"].press(*p)
    t["centre_line"].double_click(110, 15)
    t["measure"].press(0, 0); t["measure"].release(50, 0)
    t["lattice"].press(0, 45); t["lattice"].release(30, 45)                   # a1
    t["lattice"].press(2, 44); t["lattice"].release(17, 18)                   # a2, snapped to a1's start
    t["lattice"].press(60, 40); t["lattice"].release(80, 40)                  # a1 of a second lattice
    kinds = sorted(a.kind for a in session.annotations())
    assert kinds == ["centre_line", "feature", "feature", "feature", "lattice_vector", "lattice_vector",
                     "lattice_vector", "measurement_line"]
    assert all(a.label == "molecule" for a in session.annotations() if a.kind == "feature")
    vecs = {(a.attrs["lattice"], a.attrs["role"]): a for a in session.annotations() if a.kind == "lattice_vector"}
    assert set(vecs) == {("lattice 1", "a1"), ("lattice 1", "a2"), ("lattice 2", "a1")}
    a2 = vecs[("lattice 1", "a2")].geometry.points
    assert tuple(a2[0]) == pytest.approx((0, 45)) and tuple(a2[1]) == pytest.approx((15, 19))
    from scanbound.measurements import reference_axis
    assert reference_axis(session.annotations()).id == vecs[("lattice 1", "a1")].id   # a1 is the reference
    assert "reference" not in t
    assert session.store.undo() and len(session.annotations()) == 7


def test_orientation_vector_sets_its_features_orientation(session):
    from scanbound.annotations import Rect
    t = tools(session)
    t["rect"].press(10, 10); t["rect"].release(40, 30)
    feat = session.selected()[0]
    t["orientation"].press(15, 20); t["orientation"].release(35, 20)
    feat = session.store[feat.id]
    assert feat.orientation == pytest.approx(0.0)
    assert not [a for a in session.annotations() if a.kind == "orientation_vector"]   # only the centre arrow
    t["orientation"].press(25, 28); t["orientation"].release(25, 12)          # again: replaces the angle
    assert not [a for a in session.annotations() if a.kind == "orientation_vector"]
    t["orientation"].press(100, 100); t["orientation"].release(120, 100)     # off any feature: a free vector
    assert len([a for a in session.annotations() if a.kind == "orientation_vector"]) == 1
    assert session.store[feat.id].orientation == pytest.approx(90.0, abs=1.0)


def test_select_move_reshape_band_and_delete(session):
    from scanbound.annotations import Annotation, Polygon, Rect
    iid = session.current_id
    a = Annotation(iid, "feature", Rect(10, 10, 10, 10))
    b = Annotation(iid, "feature", Polygon([[50, 10], [60, 10], [60, 20]]))
    session.store.add([a, b])
    t = tools(session)["select"]
    t.press(15, 15); t.move(20, 15); t.release(20, 15)
    assert session.store[a.id].geometry == Rect(15, 10, 10, 10)
    t.press(25, 20); t.move(30, 25); t.release(30, 25)                        # the corner handle
    assert session.store[a.id].geometry == Rect(15, 10, 15, 15)
    t.press(0, 0); t.move(70, 30); t.release(70, 30)                          # rubber band
    assert set(session.selection) == {a.id, b.id}
    t.press(55, 10, mods=frozenset()); t.release(55, 10)                      # click on b's edge
    t.double_click(55, 10)
    assert len(session.store[b.id].geometry.points) == 4                     # vertex added
    t.key("delete")
    assert b.id not in session.store


def test_canvas_and_main_window_run(qapp, session, tmp_path):
    from scanbound.app.mainwindow import MainWindow
    w = MainWindow(session)
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    c = w.canvas
    assert c.image is not None and c.aspect == pytest.approx(session.image().pixel_size[1] / session.image().pixel_size[0])
    t = c.tools["rect"]
    t.press(10, 10); t.move(30, 20); t.release(30, 20)
    qapp.processEvents()
    assert len(c.layer.items) == 1
    session.store.undo()
    qapp.processEvents()
    assert len(c.layer.items) == 0
    w.save()
    assert os.path.isfile(session.project.path)
    session.project.dirty = False
    w.close()


def test_drawing_tool_deletes_only_hovered_shapes_of_its_own_type(session):
    from scanbound.annotations import Annotation, Ellipse, Polygon, Rect
    iid = session.current_id
    box = Annotation(iid, "feature", Rect(10, 10, 20, 20))
    ell = Annotation(iid, "feature", Ellipse(60, 10, 20, 20))
    poly = Annotation(iid, "feature", Polygon([[100, 10], [120, 10], [120, 30]]))
    session.store.add([box, ell, poly])
    t = tools(session)
    assert not t["rect"].delete_at(70, 20)                # an ellipse is not the box tool's shape
    assert not t["rect"].delete_at(5, 45)                 # nothing under the cursor
    assert t["rect"].delete_at(20, 20) and box.id not in session.store
    assert t["ellipse"].delete_at(70, 20) and ell.id not in session.store
    t["polygon"].press(130, 40)                           # while drawing, Delete does not delete
    assert not t["polygon"].delete_at(115, 15)
    t["polygon"].cancel()
    assert t["polygon"].delete_at(115, 15) and poly.id not in session.store
    assert not t["select"].delete_at(0, 0)
    session.store.undo()
    assert poly.id in session.store


def test_scan_limit_cancel_and_folder_tree(qapp, tmp_path):
    from scanbound.app.docks.browser import BrowserDock
    from scanbound.app.session import Session
    from scanbound.io.data import ScanStopped, list_data_files
    for d in ("a", "a/deep", "b"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        for k in range(3):
            shutil.copy(os.path.join(DATA, "bias_spec.dat"), tmp_path / d / f"s{k}.dat")
        (tmp_path / d / "notes.txt").write_text("not data")
    assert len(list_data_files(str(tmp_path), True)) == 9
    assert len(list_data_files(str(tmp_path), True, limit=4)) == 4
    seen = []
    list_data_files(str(tmp_path), True, progress=lambda d, n: seen.append((d, n)))
    assert seen[-1] == (4, 9)
    with pytest.raises(ScanStopped):
        list_data_files(str(tmp_path), True, cancelled=lambda: True)
    s = Session()
    s.open_folder(str(tmp_path), recursive=True)
    b = BrowserDock(s)
    b.rebuild()
    assert set(b.folders) == {"a", "a/deep", "b"}
    assert b.folders["a"].text(0).endswith("(6)")              # counts include subfolders
    assert list(b.items)[:3] == ["a/deep/s0.dat", "a/deep/s1.dat", "a/deep/s2.dat"]   # folders first
    b.filter.setText("deep")
    assert b.folders["b"].isHidden() and not b.folders["a"].isHidden()


def test_browser_sorting_and_favourites(qapp, tmp_path):
    from scanbound.app.docks.browser import BrowserDock
    from scanbound.app.session import Session
    import time as _t
    for name, size in (("b.dat", 3), ("a.dat", 1), ("c.dat", 2)):
        shutil.copy(os.path.join(DATA, "bias_spec.dat"), tmp_path / name)
        with open(tmp_path / name, "ab") as fh:
            fh.write(b"\n" * size * 100)
    s = Session()
    s.open_folder(str(tmp_path))
    b = BrowserDock(s)
    b.rebuild()
    assert list(b.items) == ["a.dat", "b.dat", "c.dat"]
    b.sort.setCurrentIndex(b.sort.findData("size"))
    assert list(b.items) == ["a.dat", "c.dat", "b.dat"]
    b.sort_desc.setChecked(True)
    assert list(b.items) == ["b.dat", "c.dat", "a.dat"]
    b.set_favourite(["c.dat"], True)
    assert s.project.settings["favourites"] == ["c.dat"] and s.project.dirty
    assert b.items["c.dat"].text(0).startswith("★ ")
    b.only_fav.setChecked(True)
    assert [r for r, it in b.items.items() if not it.isHidden()] == ["c.dat"]
    b.rebuild()                                   # stars survive a rebuild (they live in the project)
    assert b.items["c.dat"].text(0).startswith("★ ")
    b.sort.setCurrentIndex(b.sort.findData("name")); b.sort_desc.setChecked(False)


def test_escape_deselects_with_any_tool(qapp, session):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from scanbound.annotations import Annotation, Rect
    from scanbound.app.canvas import ImageCanvas
    c = ImageCanvas(session)
    c._on_image_changed(session.current_id)
    a = Annotation(session.current_id, "feature", Rect(5, 5, 10, 10))
    session.store.add(a)
    for tool in ("rect", "polygon", "select", "measure"):
        c.set_tool(tool)
        session.select([a.id])
        QTest.keyClick(c, Qt.Key.Key_Escape)
        assert session.selection == [], tool


def test_processing_steps_stay_with_their_image(qapp, session, tmp_path):
    from scanbound.app.docks.processing import ProcessingTab
    from scanbound.processing import ChainStep, ProcessingChain
    shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / "scan2.sxm")
    session.rescan()
    a, b = session.image_ids()[:2]
    tab = ProcessingTab(session)
    session.set_current(a)
    session.set_chain(ProcessingChain([ChainStep("plane", True, {})]))
    session.set_current(b)
    session.set_current(a)
    assert [c.spec.key for c in tab.panel.active()] == ["plane"]      # the panel shows A's steps again
    tab.keep.setChecked(True)
    session.set_current(b)
    assert session.project.entry(b).chain.summary() == session.project.entry(a).chain.summary()
    tab.keep.setChecked(False)


def test_results_are_accepted_and_reruns_replace_untouched_ones(session):
    from scanbound.annotations import Annotation, Origin, Rect
    from scanbound.app.dialogs.detect import _add_results
    from scanbound.app.features import REVIEW
    assert not REVIEW                                                  # the review workflow is switched off
    iid = session.current_id
    origin = Origin("detector", "adapt pipeline")

    def run(x):  # noqa: ANN001, ANN202
        res = {iid: [Annotation(iid, "feature", Rect(x, 1, 5, 5), status="proposed", origin=origin)]}
        _add_results(session, res, "detector", True, "test")

    run(1)
    assert [a.status for a in session.annotations()] == ["accepted"]  # no proposals
    run(9)                                                             # replaces the untouched result
    (a,) = session.annotations()
    assert a.geometry == Rect(9, 1, 5, 5) and a.status == "accepted"
    session.store.move([a.id], 1, 0)                                   # an edited result is kept
    run(20)
    assert sorted(a.geometry.x for a in session.annotations()) == [10, 20]
    session.store.undo()
    assert len(session.annotations()) == 1


def test_proposals_in_an_opened_project_are_accepted(qapp, tmp_path):
    from scanbound.annotations import Annotation, Origin, Rect
    from scanbound.app.session import Session
    from scanbound.project import Project
    shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / "scan.sxm")
    p = Project.open_folder(str(tmp_path))
    iid = p.entry("scan.sxm::").id
    p.annotations.add(Annotation(iid, "feature", Rect(1, 1, 5, 5), status="proposed", origin=Origin("detector")))
    s = Session(p)
    assert [a.status for a in s.store] == ["accepted"] and not s.store.can_undo


def test_publication_figure_formats_and_layout(tmp_path):
    from scanbound.io import open_data
    from scanbound.publication import FigureStyle, figure_png, render_figure, save_figure
    img = open_data(os.path.join(DATA, "aniso_0020.sxm")).image()
    for st in (FigureStyle(), FigureStyle(colorbar="bottom", axes=True, panel_label="b", dpi=150),
               FigureStyle(colorbar="none", crop=(0, 0, 40, 20), width_mm=60)):
        fig = render_figure(img, None, st)
        assert abs(fig.get_size_inches()[0] - st.width_mm / 25.4) < 1e-6
        for ext in (".png", ".pdf", ".svg", ".tif"):
            p = save_figure(fig, str(tmp_path / f"f{ext}"), st, image=img)
            assert os.path.getsize(p) > 1000 and os.path.isfile(p + ".json")
    png = figure_png(render_figure(img, None, FigureStyle(dpi=300)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    h, w = img.shape
    dx, dy = img.pixel_size                           # the picture keeps the physical aspect, not the pixel grid
    fig = render_figure(img, None, FigureStyle(colorbar="none", scale_bar=False))
    W, H = fig.get_size_inches()
    assert H / W == pytest.approx(h * dy / (w * dx), rel=1e-3)
    with pytest.raises(ValueError):
        save_figure(fig, str(tmp_path / "f.bmp"))


def test_publication_figure_text_always_fits(tmp_path):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scanbound.io import open_data
    from scanbound.publication import FigureStyle, render_figure
    img = open_data(os.path.join(DATA, "aniso_0020.sxm")).image()
    for cb in ("right", "bottom", "none"):
        for axes in (True, False):
            for size, width in ((6, 60), (8, 85), (12, 85), (10, 175)):
                st = FigureStyle(colorbar=cb, axes=axes, font_size=size, width_mm=width,
                                 colorbar_label="Apparent height above the substrate (pm)")
                fig = render_figure(img, None, st)
                with mpl.rc_context(fig._scanbound_rc):         # as figure_png / save_figure draw it
                    FigureCanvasAgg(fig)
                    r = fig.canvas.get_renderer()
                    W, H = fig.get_size_inches()
                    box = fig.get_tightbbox(r)
                assert box.x0 > -0.01 and box.y0 > -0.01 and box.x1 < W + 0.01 and box.y1 < H + 0.01, (cb, axes, size)
                if cb == "bottom" and axes:              # the x title stays clear of the colour bar
                    ax, cax = fig.axes[0], fig.axes[-1]
                    assert ax.get_tightbbox(r).y0 > cax.get_window_extent(r).y1
    bare = render_figure(img, None, FigureStyle(colorbar="none", scale_bar=False))
    assert bare.axes[0].get_position().bounds == pytest.approx((0, 0, 1, 1))   # a bare image has no margin


def test_ticked_spectra_get_dots_in_their_plot_colour(qapp, session):
    import dataclasses
    import numpy as np
    from PyQt6.QtCore import Qt
    from scanbound.app.docks.spectra import SpectraDock
    dock = SpectraDock(session)
    dock.rebuild()
    img = session.image()
    h, w = img.shape
    rel = "spec.dat"
    world = img.frame.pixel_to_world(np.array([[w / 2, h / 2]]), img.shape)[0]
    dock._cache[rel] = dataclasses.replace(dock.spectrum(rel), position=(float(world[0]), float(world[1]), 0.0))
    got = []
    dock.positionsChanged.connect(got.append)
    dock.show_file(rel)                               # as double-clicking it in the file list does
    (r, x, y, color), = got[-1]
    assert r == rel and color == dock.ax.get_lines()[0].get_color()
    assert x == pytest.approx(w / 2, abs=0.01) and y == pytest.approx(h / 2, abs=0.01)
    dock.files.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert len(got[-1]) == 1                          # still plotted (the highlighted one), so still marked
    dock.files.setCurrentItem(None)
    dock.update_positions()
    assert got[-1] == []                              # nothing plotted, 'all positions' off: no marker


def test_theme_switch_and_tool_icons(qapp):
    from PyQt6.QtWidgets import QLabel
    from scanbound.app.icons import PAINTERS, tool_icon
    from scanbound.app.theme import apply_theme, mark, tone
    from scanbound.app.tools import TOOLS
    assert set(PAINTERS) == set(TOOLS)
    assert all(not tool_icon(k).isNull() for k in TOOLS)
    style = qapp.style().name()
    lab = mark(QLabel("x"), "muted")
    try:
        apply_theme("dark")
        assert lab.palette().windowText().color().name() == tone("muted")
        apply_theme("light")
        assert lab.palette().windowText().color().name() == tone("muted")
        assert qapp.style().name() == style and qapp.styleSheet() == ""    # the native look is kept
    finally:
        apply_theme("light")


def test_chosen_channel_is_kept_across_files(qapp, tmp_path):
    from scanbound.app.session import Session
    for name in ("a.sxm", "b.sxm"):
        shutil.copy(os.path.join(DATA, "aniso_0020.sxm"), tmp_path / name)
    from PIL import Image as PILImage
    PILImage.new("L", (8, 8)).save(tmp_path / "c.png")           # a file without that channel
    s = Session()
    s.open_folder(str(tmp_path))
    s.set_current("a.sxm::")
    assert s.current_id == "a.sxm::Z"
    s.set_channel("Current")
    assert s.current_id == "a.sxm::Current"
    s.set_current("b.sxm::")
    assert s.current_id == "b.sxm::Current"                       # kept
    s.set_current("c.png::")
    rel, ch = s.project.split_image_id(s.current_id)
    assert rel == "c.png" and ch == s.project.data_file("c.png").default_channel   # back to the default
    s.set_current("a.sxm::")
    assert s.current_id == "a.sxm::Current"                       # still remembered


def test_label_names_toggle_keeps_items_and_measure_opens_profiles(qapp, session):
    from scanbound.app.mainwindow import MainWindow
    w = MainWindow(session)
    w.show()
    c = w.canvas
    t = c.tools["rect"]
    for i in range(20):                                                    # objects without a label
        t.press(2 + 6 * i % 130, 5); t.move(6 + 6 * i % 130, 9); t.release(6 + 6 * i % 130, 9)
    qapp.processEvents()
    items = dict(c.layer.items)
    for _ in range(4):
        w.a_labels.trigger()
        qapp.processEvents()
        c.viewport().grab()
    assert c.layer.items == items                                          # restyled, not rebuilt
    assert w._profile_window is None
    c.set_tool("measure")
    assert w._profile_window is not None and w._profile_window.isVisible()
    session.project.dirty = False
    w._profile_window.close()
    w.close()
