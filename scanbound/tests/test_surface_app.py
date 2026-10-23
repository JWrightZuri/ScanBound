"""Surface panel, overlay, report windows and the Surface menu, offscreen on a synthetic lattice."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtCore = pytest.importorskip("PyQt6.QtCore")
cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _wait(qapp, cond, timeout=120.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    assert cond()


def test_surface_panel_end_to_end(qapp, tmp_path):
    from scanbound.annotations import Annotation, Point
    from scanbound.app.dialogs.surface_report import ReportWindow
    from scanbound.app.docks.surface import MAPS
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.project import Project
    from scanbound.surface import workflow as wf
    from scanbound.surface.store import SurfaceStore
    from scanbound.surface.structure import preset
    from scanbound.surface.synthetic import lattice_image

    img, truth = lattice_image(size=(200, 200), vacancies=0.02, adatoms=20, seed=3)
    a = np.asarray(img.data)
    cv2.imwrite(str(tmp_path / "s1.png"), ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16))
    s = Session()
    w = MainWindow(s)
    s.set_project(Project.open_folder(str(tmp_path)))
    p = s.project
    iid = s.current_id
    s.set_calibration(*truth["pixel"])
    for k, xy in enumerate(truth["adatoms"]):
        p.annotations.add(Annotation(iid, "feature", Point(*xy), label="mol", orientation=float(k % 2 * 60)))
    reports = []
    panel = w.surface
    panel.reportReady.disconnect()
    panel.reportReady.connect(reports.append)

    panel.fit_lattice()
    _wait(qapp, lambda: not panel._busy)
    assert SurfaceStore(p).fit(iid) is not None and "a1 =" in panel.result.text()
    assert w.surface_overlay.fit is not None

    SurfaceStore(p).set_model(preset("fcc(111)", "Cu"))
    panel._fill_models()
    panel._model_changed()
    assert panel.align.count() >= 1 and panel.origin.count() >= 1
    panel.use_alignment()
    assert SurfaceStore(p).alignment(iid) is not None and w.surface_overlay._atoms
    for key in ("dilatation", "domains"):
        panel.map.setCurrentIndex([k for k, _ in MAPS].index(key))
        assert w.surface_overlay.map_item.isVisible()
    panel.map.setCurrentIndex(0)

    for fn in (wf.structure_explorer, wf.adsorption_map, wf.defect_map, wf.quality_check):
        n = len(reports)
        panel.run(fn)
        _wait(qapp, lambda: not panel._busy)
        assert len(reports) == n + 1, panel.status.text()
    for r in reports:
        rw = ReportWindow(s, r, w, on_proposals=w.add_proposals)
        rw.show()
        qapp.processEvents()
        rw.close()
    n = len(p.annotations)
    w.add_proposals(reports[2].proposals)
    assert len(p.annotations) == n + len(reports[2].proposals)

    mol = p.annotations.for_image(iid, labels=["mol"])[0]
    s.select([mol.id])
    w.set_handedness("R")
    assert p.annotations[mol.id].attrs["handedness"] == "R"
    panel.explain()
    _wait(qapp, lambda: not panel._busy)
    assert reports[-1].tables["hypotheses"]

    w.canvas.set_zoom(6.0)
    w.canvas.viewport().repaint()
    assert not w.canvas.grab().isNull()
    p.save()
    w.close()


def test_builder_window(qapp, tmp_path, monkeypatch):
    from scanbound.annotations import Annotation, Point
    from scanbound.app.dialogs import builder3d
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.project import Project
    from scanbound.surface.builder import read_xyz
    from scanbound.surface.store import SurfaceStore
    from scanbound.surface.synthetic import lattice_image

    img, truth = lattice_image(size=(160, 160), adatoms=6, seed=4)
    a = np.asarray(img.data)
    cv2.imwrite(str(tmp_path / "s1.png"), ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16))
    xyz = tmp_path / "rod.xyz"
    xyz.write_text("4\nrod\nC 0 0 0\nC 1.5 0 0\nC 3.0 0 0\nH 3.8 1.0 0\n")
    s = Session()
    w = MainWindow(s)
    s.set_project(Project.open_folder(str(tmp_path)))
    p = s.project
    iid = s.current_id
    panel = w.surface
    messages = []
    monkeypatch.setattr(builder3d.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(xyz)], ""))
    from scanbound.app.docks import surface as surface_dock
    monkeypatch.setattr(surface_dock.QMessageBox, "information", lambda *a: messages.append(a[2]))

    panel.build_model()                              # nothing prepared yet: says what is missing
    assert messages and "fit the lattice" in messages[-1] and "molecule annotations" in messages[-1]

    s.set_calibration(*truth["pixel"])
    for k, xy in enumerate(truth["adatoms"]):
        p.annotations.add(Annotation(iid, "feature", Point(*xy), label="mol", orientation=float(30 * k)))
    panel.fit_lattice()
    _wait(qapp, lambda: not panel._busy)
    n = len(messages)
    panel.build_model()
    assert len(messages) == n
    win = panel._builders[iid]
    assert "rod" in win.templates and win.scene is not None
    assert win.table.rowCount() == len(truth["adatoms"])

    win.select_ids([win.scene.placements[0].id])
    win.e_height.setValue(4.25)
    win.e_dx.setValue(1.0)
    pl = win.scene.placements[0]
    assert pl.height == pytest.approx(4.25) and pl.x == pytest.approx(pl.x0 + 1.0)
    assert win.scene.molecule(pl)[1][:, 2].min() == pytest.approx(4.25)
    win._set_all("height", 3.0)
    assert all(q.height == 3.0 for q in win.scene.placements)
    win.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not win.scene.placements[1].enabled
    win.reset_selected()
    qapp.processEvents()
    win.redraw()

    out = tmp_path / "out.xyz"
    nat = win.export_to(str(out))
    syms, pos, _ = read_xyz(str(out))
    assert len(syms) == nat == len(win.scene.substrate_symbols) + 4 * (len(truth["adatoms"]) - 1)
    win.close()
    st = SurfaceStore(p)
    assert st.data["builder"]["label_map"] == {"mol": "rod"}
    p.save()
    w.close()


def test_panel_preview_and_molecule_files(qapp, tmp_path):
    from scanbound.annotations import Annotation, Point
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.project import Project
    from scanbound.surface.builder import load_molecules
    from scanbound.surface.store import SurfaceStore
    from scanbound.surface.synthetic import lattice_image

    img, truth = lattice_image(size=(160, 160), adatoms=6, seed=5)
    a = np.asarray(img.data)
    cv2.imwrite(str(tmp_path / "s1.png"), ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16))
    for name, el in (("rod", "C"), ("dot", "N")):
        (tmp_path / f"{name}.xyz").write_text(f"3\n{name}\n{el} 0 0 0\n{el} 1.4 0 0\n{el} 2.8 0 0\n")
    s = Session()
    w = MainWindow(s)
    s.set_project(Project.open_folder(str(tmp_path)))
    p = s.project
    iid = s.current_id
    s.set_calibration(*truth["pixel"])
    for k, xy in enumerate(truth["adatoms"]):
        p.annotations.add(Annotation(iid, "feature", Point(*xy), label="rod" if k % 2 else "dot"))
    panel = w.surface
    qapp.processEvents()

    # Display is the first tab and the one shown
    w.show()
    qapp.processEvents()
    assert w.tabifiedDockWidgets(w.docks["Display"])
    assert not w.docks["Display"].visibleRegion().isEmpty() and w.docks["Labels"].visibleRegion().isEmpty()
    bars = [tb for tb in w.findChildren(QtWidgets.QTabBar) if tb.count() == 5]    # no Review panel
    assert bars and bars[0].tabText(0) == "Display" and bars[0].currentIndex() == 0

    # fine spacing / threshold steps, estimate and live preview
    panel._update_spacing_unit()
    assert panel.spacing.singleStep() == pytest.approx(0.01) and panel.spacing.suffix().strip() == "nm"
    assert panel.threshold.singleStep() == pytest.approx(0.05)
    panel.estimate_spacing()
    assert panel.spacing.value() == pytest.approx(0.28, rel=0.05)
    panel.preview.setChecked(True)
    _wait(qapp, lambda: panel._candidates is not None and not panel._preview_busy)
    _wait(qapp, lambda: len(w.surface_overlay._preview) > 0)
    n_hi = int(w.surface_overlay._preview_on.sum())
    panel.thr_slider.setValue(50)
    assert panel.threshold.value() == pytest.approx(0.5)
    assert int(w.surface_overlay._preview_on.sum()) >= n_hi and "sites used" in panel.preview_label.text()
    panel.threshold.setValue(1.5)
    assert panel.thr_slider.value() == 150
    panel.fit_lattice()
    _wait(qapp, lambda: not panel._busy)
    assert not panel.preview.isChecked() and len(w.surface_overlay._preview) == 0

    # molecule files in the panel, auto-assigned by name, used by the 3D window; first through the button
    from scanbound.app.docks import surface as surface_dock
    picked = [str(tmp_path / "rod.xyz")]
    orig = surface_dock.QFileDialog.getOpenFileNames
    surface_dock.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (picked, ""))
    try:
        add_btn = next(b for b in panel.findChildren(QtWidgets.QPushButton) if b.text() == "Add .xyz…")
        add_btn.click()
    finally:
        surface_dock.QFileDialog.getOpenFileNames = orig
    assert panel.mol_list.count() == 1
    panel.mol_list.setCurrentRow(0)
    panel.remove_molecule()
    assert panel.mol_list.count() == 0
    panel.add_molecules([str(tmp_path / "rod.xyz"), str(tmp_path / "dot.xyz")])
    temps, label_map = load_molecules(SurfaceStore(p))
    assert set(temps) == {"rod", "dot"} and label_map == {"dot": "dot", "rod": "rod"}
    assert panel.mol_list.count() == 2 and panel.mol_map.rowCount() == 2
    panel._assign("dot", "rod")
    assert load_molecules(SurfaceStore(p))[1]["dot"] == "rod"
    panel._assign("dot", "dot")
    panel.build_model()
    win = panel._builders[iid]
    assert {q.template for q in win.scene.placements} == {"rod", "dot"}
    panel.mol_list.setCurrentRow(list(temps).index("dot"))
    panel.remove_molecule()
    assert set(win.templates) == {"rod"} and {q.template for q in win.scene.placements} == {"rod"}
    win.close()
    p.save()
    w.close()


def test_feedback_for_failed_fit_and_model(qapp, tmp_path, monkeypatch):
    from scanbound.app.docks import surface as surface_dock
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.project import Project
    from scanbound.surface.store import SurfaceStore
    from scanbound.surface.structure import preset
    from scanbound.surface.synthetic import lattice_image

    rng = np.random.default_rng(0)
    cv2.imwrite(str(tmp_path / "flat.png"), (np.full((120, 120), 30000) + rng.integers(0, 3, (120, 120))).astype(np.uint16))
    img, truth = lattice_image(size=(160, 160), seed=6)
    a = np.asarray(img.data)
    cv2.imwrite(str(tmp_path / "s1.png"), ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16))
    shown = []
    monkeypatch.setattr(surface_dock.QMessageBox, "warning", lambda *a: shown.append(("warning", a[2])))
    monkeypatch.setattr(surface_dock.QMessageBox, "information", lambda *a: shown.append(("info", a[2])))
    s = Session()
    w = MainWindow(s)
    s.set_project(Project.open_folder(str(tmp_path)))
    panel = w.surface
    ids = sorted(s.project.image_ids())
    flat = next(i for i in ids if "flat" in i)
    lat = next(i for i in ids if "s1" in i)

    # a fit that fails says why, under the button and in a dialog with what to try
    s.set_current(flat)
    panel.threshold.setValue(15.0)
    panel.fit_lattice()
    _wait(qapp, lambda: not panel._busy)
    assert shown and shown[-1][0] == "warning" and "What to try" in shown[-1][1]
    assert panel.result.text().startswith("The lattice fit failed") and panel.result.property("tone") == "error"

    # Use without a fit / model, and a model that does not fit, are explained
    s.set_current(lat)
    s.set_calibration(*truth["pixel"])
    panel.use_alignment()
    assert "model" in shown[-1][1].lower()
    SurfaceStore(s.project).set_model(preset("fcc(100)", "Cu"))      # square cell on a hexagonal lattice
    panel._fill_models()
    panel._model_changed()
    assert "Fit the lattice" in panel.align_label.text()
    panel.threshold.setValue(1.5)
    panel.fit_lattice()
    _wait(qapp, lambda: not panel._busy)
    assert panel.result.property("tone") in ("ok", "warn")
    panel._model_changed()
    assert "does not match the fitted lattice" in panel.align_label.text() and "angle off" in panel.align_label.text()
    n = len(shown)
    panel.use_alignment()
    assert len(shown) == n + 1 and "does not match" in shown[-1][1]
    SurfaceStore(s.project).set_model(preset("fcc(111)", "Cu"))
    panel._fill_models()
    panel.model.setCurrentText("Cu(111)") if panel.model.findText("Cu(111)") >= 0 else None
    panel._model_changed()
    if panel._alignments:
        panel.use_alignment()
        assert panel.align_label.text().startswith("In use") and "scale 1.09" in panel.align_label.text()
        assert "differs from the model's by +10%" in panel.align_label.text()   # synthetic 0.28 nm vs Cu 0.256 nm
    s.project.save()
    w.close()
