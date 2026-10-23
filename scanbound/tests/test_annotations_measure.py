"""Annotation model, store, legacy format and measurements: geometry edge cases and units."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scanbound.annotations import (Annotation, AnnotationStore, Ellipse, Point, Polygon, Polyline, Rect, Segment,
                                   validate)
from scanbound.annotations.kinds import ValidationContext
from scanbound.annotations.legacy import from_frame, to_frame
from scanbound.measurements import lattice, measure_annotation, measure_image
from scanbound.model import Image
from scanbound.model.axes import Axis

IMG = "scan.sxm::Z"


def calibrated(shape=(100, 200), dx=1e-10, dy=2e-10, rel=0.02) -> Image:
    from scanbound.model.axes import Calibration
    cal = Calibration("file", "test", rel)
    h, w = shape
    return Image(np.zeros(shape), x=Axis.spatial("x", w, dx, calibration=cal), y=Axis.spatial("y", h, dy, calibration=cal),
                 quantity="Z", unit="m", id=IMG)


# --------------------------------------------------------------------------- store
def test_close_annotations_hit_test_picks_the_nearest_and_smallest():
    s = AnnotationStore()
    big = Annotation(IMG, "feature", Rect(0, 0, 50, 50))
    small = Annotation(IMG, "feature", Rect(20, 20, 2, 2))        # inside the big one
    a = Annotation(IMG, "feature", Rect(60, 0, 10, 10))
    b = Annotation(IMG, "feature", Rect(70.3, 0, 10, 10))         # 0.3 px gap
    s.add([big, small, a, b])
    assert s.hit_test(IMG, 21, 21)[0].id == small.id
    assert s.hit_test(IMG, 10, 10)[0].id == big.id
    assert s.hit_test(IMG, 69.9, 5, tolerance=1)[0].id == a.id
    assert s.hit_test(IMG, 70.4, 5, tolerance=1)[0].id == b.id


def test_transactions_undo_as_one_step_and_roll_back_on_error():
    s = AnnotationStore()
    events = []
    s.subscribe(events.append)
    a = Annotation(IMG, "feature", Rect(0, 0, 5, 5), status="proposed")
    s.add(a)
    with s.transaction("Review"):
        s.set_label(a.id, "molecule")
        s.accept(a.id)
    assert s[a.id].status == "accepted" and s[a.id].label == "molecule" and s[a.id].edited
    assert len(events) == 2
    s.undo()
    assert s[a.id].status == "proposed" and s[a.id].label is None
    s.redo()
    assert s[a.id].status == "accepted"
    with pytest.raises(RuntimeError):
        with s.transaction("Broken"):
            s.set_label(a.id, "other")
            raise RuntimeError("boom")
    assert s[a.id].label == "molecule"                             # rolled back


def test_single_id_is_not_split_into_characters():
    s = AnnotationStore()
    a = Annotation(IMG, "feature", Rect(0, 0, 5, 5), status="proposed")
    s.add(a)
    s.reject(a.id)
    assert s[a.id].status == "rejected"
    s.remove(a.id)
    assert a.id not in s


def test_validation_rules():
    s = AnnotationStore()
    ref1 = Annotation(IMG, "reference_axis", Segment([[0, 0], [20, 0]]))
    ref2 = Annotation(IMG, "reference_axis", Segment([[0, 5], [20, 5]]))
    out = Annotation(IMG, "feature", Rect(190, 90, 30, 30))
    lone = Annotation(IMG, "lattice_vector", Segment([[0, 0], [5, 0]]), attrs={"lattice": "L"})
    s.add([ref1, ref2, out, lone])
    ctx = ValidationContext((100, 200), s)
    codes = {i.code for a in (ref2, out, lone) for i in validate(a, ctx)}
    assert "not_unique" in codes
    assert any("outside" in c for c in codes)
    assert any("pair" in c or "lattice" in c for c in codes)


def test_geometry_round_trip_and_degenerate_shapes():
    for g in (Point(1, 2), Segment([[0, 0], [3, 4]]), Polyline([[0, 0], [1, 1], [2, 0]]),
              Polygon([[0, 0], [4, 0], [4, 3]]), Rect(1, 2, 3, 4), Ellipse(0, 0, 4, 2)):
        a = Annotation(IMG, "feature" if g.closed or g.type == "point" else "centre_line", g)
        b = Annotation.from_dict(a.to_dict())
        assert b.geometry == g and b.kind == a.kind
    assert Rect(0, 0, -3, -2) == Rect(-3, -2, 3, 2)
    with pytest.raises(ValueError):
        Rect(0, 0, float("nan"), 1)
    assert Polygon([[0, 0], [1, 1], [2, 2]]).is_degenerate()


# --------------------------------------------------------------------------- legacy CSV
def test_legacy_csv_round_trip_keeps_shapes_and_labels():
    rows = [
        ["r1", "RectROI", 10, 10, 5, 6, "a.jpg", "mol", np.nan],
        ["p1", "PolygonROI", 0, 0, 0, 0, "a.jpg", "island", np.nan],
        ["p1", "PolygonROI", 10, 0, 0, 0, "a.jpg", "island", np.nan],
        ["p1", "PolygonROI", 10, 10, 0, 0, "a.jpg", "island", np.nan],
        ["l1", "LineSegmentROI", 1, 1, 0, 0, "a.jpg", "GNR", np.nan],
        ["l1", "LineSegmentROI", 8, 3, 0, 0, "a.jpg", "GNR", np.nan],
        ["v1", "VectorROI", 11, 12, 14, 12, "a.jpg", "mol", np.nan],
        ["s1", "VectorROI", 0, 0, 30, 0, "a.jpg", "Symmetry Vector", np.nan],
        ["e1", "PointROI", 50, 50, 0, 0, "a.jpg", "Empty Label", np.nan],
    ]
    df = pd.DataFrame(rows, columns=["objectName", "type", "x1", "y1", "dx", "dy", "file", "label", "angle"])
    anns, problems = from_frame(df, lambda name: (IMG, None))
    assert not problems
    kinds = sorted(a.kind for a in anns)
    assert kinds == ["centre_line", "feature", "feature", "feature", "orientation_vector", "reference_axis"]
    vec = next(a for a in anns if a.kind == "orientation_vector")
    rect = next(a for a in anns if a.geometry.type == "rect")
    assert vec.parent == rect.id                                    # the vector belongs to the box it is in
    assert next(a for a in anns if a.geometry.type == "point").label is None
    back = to_frame(anns, lambda i: "a.jpg")
    assert len(back) == len(df)


# --------------------------------------------------------------------------- measurements
def test_anisotropic_lengths_and_areas_are_physical():
    img = calibrated(dx=1e-10, dy=2e-10)
    line = Annotation(IMG, "measurement_line", Segment([[0, 0], [0, 10]]))      # 10 px along y = 2 nm
    m = measure_annotation(line, img)
    assert m["length"].value == pytest.approx(2e-9)
    assert m["length"].unit == "m"
    box = Annotation(IMG, "feature", Rect(0, 0, 10, 10))
    mb = measure_annotation(box, img)
    assert mb["area"].value == pytest.approx(10 * 1e-10 * 10 * 2e-10)
    diag = Annotation(IMG, "measurement_line", Segment([[0, 0], [20, 10]]))       # 2 nm right, 2 nm down
    assert measure_annotation(diag, img)["angle"].value == pytest.approx(135.0)     # a line has no head: 0-180°


def test_uncertainty_includes_calibration_and_placement():
    img = calibrated(dx=1e-10, dy=1e-10, rel=0.02)
    line = Annotation(IMG, "measurement_line", Segment([[0, 0], [100, 0]]))
    m = measure_annotation(line, img)["length"]
    assert m.value == pytest.approx(1e-8)
    assert m.uncertainty >= 0.02 * 1e-8 * 0.99                       # at least the calibration part
    assert m.uncertainty < 0.05 * 1e-8
    text = m.explain()
    assert "Uncertainty" in text and "Assumes" in text


def test_uncalibrated_image_measures_in_pixels():
    img = Image(np.zeros((50, 50)), id=IMG)
    m = measure_annotation(Annotation(IMG, "measurement_line", Segment([[0, 0], [3, 4]])), img)["length"]
    assert m.value == pytest.approx(5.0) and m.unit == "px"


def test_relative_angle_and_lattice():
    img = calibrated(dx=1e-10, dy=1e-10)
    ref = Annotation(IMG, "reference_axis", Segment([[0, 0], [10, -10]]))           # 45°
    vec = Annotation(IMG, "orientation_vector", Segment([[0, 0], [0, -10]]))        # 90°
    m = measure_annotation(vec, img, reference=ref)
    assert m["relative_angle"].value == pytest.approx(45.0)
    a1 = Annotation(IMG, "lattice_vector", Segment([[0, 0], [25, 0]]), attrs={"lattice": "L"})
    a2 = Annotation(IMG, "lattice_vector", Segment([[0, 0], [12.5, -21.65]]), attrs={"lattice": "L"})
    lat = lattice(a1, a2, img)
    assert lat["a1"].value == pytest.approx(2.5e-9)
    assert lat["a2"].value == pytest.approx(2.5e-9, rel=1e-3)
    assert lat["gamma"].value == pytest.approx(60.0, abs=0.05)
    rows = measure_image([ref, vec, a1, a2], img)
    assert any(r["measurement"].startswith("L ") for r in rows)
