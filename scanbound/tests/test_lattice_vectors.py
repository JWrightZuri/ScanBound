"""Lattice vectors typed in as numbers: the model helpers, the dialog, and a fit started from them."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from scanbound.annotations import Annotation, AnnotationStore
from scanbound.annotations import lattice_vectors as lv
from scanbound.annotations.geometry import Segment
from scanbound.measurements.report import lattice_pairs, reference_axis


def test_values_round_trip_with_non_square_pixels():
    scale = (0.02e-9, 0.04e-9)
    v = lv.LatticeValues(0.288e-9, 15.0, 0.288e-9, 75.0, (50.0, 60.0), "m")
    assert v.gamma == pytest.approx(60.0)
    s1, s2 = lv.segments(v, scale)
    assert s1.points[0] == pytest.approx((50.0, 60.0)) and s2.points[0] == pytest.approx((50.0, 60.0))
    # 15° CCW (up) means y decreases; physical length is kept on anisotropic pixels
    d = np.subtract(s1.points[1], s1.points[0]) * scale
    assert d[1] < 0 and math.hypot(*d) == pytest.approx(0.288e-9)
    anns = [Annotation("im", "lattice_vector", s, attrs={"lattice": "lattice 1", "role": r})
            for s, r in ((s1, "a1"), (s2, "a2"))]
    back = lv.values_from_pair(anns, scale, "m")
    assert back.a1_angle == pytest.approx(15.0) and back.gamma == pytest.approx(60.0)
    assert back.a2_length == pytest.approx(0.288e-9)
    a1, a2, name = lv.physical_pair(anns, scale)
    assert name == "lattice 1" and np.allclose(a1, v.physical()[0]) and np.allclose(a2, v.physical()[1])
    assert v.swapped().gamma == pytest.approx(-60.0)
    assert lv.LatticeValues(1, 0, 1, 179, (0, 0)).problems() and lv.LatticeValues(0, 0, 1, 60, (0, 0)).problems()


def test_typed_values_update_drawn_vectors_in_place():
    store = AnnotationStore()
    a1 = Annotation("im", "lattice_vector", Segment([(10, 10), (30, 10)]), attrs={"lattice": "lattice 1", "role": "a1"})
    store.add(a1)
    v = lv.LatticeValues(25.0, 0.0, 25.0, 120.0, (10.0, 10.0))
    new, updated = lv.annotations_for(v, "im", (1.0, 1.0), "lattice 1", store.for_image("im"))
    assert [u.id for u in updated] == [a1.id] and len(new) == 1 and new[0].attrs["role"] == "a2"
    with store.transaction("typed"):
        for u in updated:
            store.replace(u)
        store.add(new)
    pair = lattice_pairs(store.for_image("im"))["lattice 1"]
    assert [p.attrs["role"] for p in pair] == ["a1", "a2"] and pair[0].geometry.length() == pytest.approx(25.0)
    assert reference_axis(store.for_image("im")).id == a1.id
    store.undo()
    assert len(store.for_image("im")) == 1


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_dialog_and_fit_from_typed_vectors(tmp_path):
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    cv2 = pytest.importorskip("cv2")
    from scanbound.app.dialogs.surface_tools import LatticeVectorsDialog
    from scanbound.app.mainwindow import MainWindow
    from scanbound.app.session import Session
    from scanbound.project import Project
    from scanbound.surface.synthetic import lattice_image
    import time

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    img, truth = lattice_image(size=(200, 200), angle=12, seed=3)
    a = np.asarray(img.data)
    cv2.imwrite(str(tmp_path / "s1.png"), ((a - a.min()) / np.ptp(a) * 65535).astype(np.uint16))
    s = Session()
    w = MainWindow(s)
    s.set_project(Project.open_folder(str(tmp_path)))
    iid = s.current_id
    s.set_calibration(*truth["pixel"])
    a_nm = float(np.hypot(*truth["a1"])) * 1e9

    dlg = LatticeVectorsDialog(s, iid, None)
    assert dlg.lattice.currentText() == dlg.NEW and dlg.l1.suffix() == " nm"
    dlg.l1.setValue(a_nm * 1.03)                      # a rough guess is enough
    dlg.l2.setValue(a_nm * 1.03)
    dlg.t1.setValue(12.0)
    dlg.gamma.setValue(60.0)
    dlg.gamma.setValue(1.0)
    assert dlg.apply() and not s.store.for_image(iid, kinds=("lattice_vector",))     # parallel: refused
    dlg.gamma.setValue(60.0)
    assert dlg.apply() == []
    vecs = s.store.for_image(iid, kinds=("lattice_vector",))
    assert sorted(v.attrs["role"] for v in vecs) == ["a1", "a2"] and all(v.attrs["lattice"] == "lattice 1" for v in vecs)

    # editing the same lattice again changes the vectors, not their number
    dlg2 = LatticeVectorsDialog(s, iid, None, "lattice 1")
    assert dlg2.l1.value() == pytest.approx(a_nm * 1.03, rel=1e-3) and dlg2.gamma.value() == pytest.approx(60.0, abs=0.01)
    dlg2.l1.setValue(a_nm)
    dlg2.l2.setValue(a_nm)
    assert dlg2.apply() == []
    assert len(s.store.for_image(iid, kinds=("lattice_vector",))) == 2

    panel = w.surface
    assert panel.from_vectors.isChecked()
    panel.fit_lattice()
    t0 = time.time()
    while panel._busy and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert not panel._busy
    fit = panel.store.fit(iid)
    assert fit is not None and "lattice vectors of lattice 1" in panel.result.text()
    assert float(np.hypot(*fit.a1)) == pytest.approx(a_nm * 1e-9, rel=0.02)
    from_fit = LatticeVectorsDialog(s, iid, fit, dlg.NEW)       # a new lattice starts from the fit
    assert from_fit.l1.value() == pytest.approx(float(np.hypot(*fit.a1)) * 1e9, rel=1e-3)
    s.project.save()
    w.close()
