"""Dataset export, evaluation, review statistics and analyses on small synthetic projects."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image as PILImage

from scanbound.analysis import analysis_context
from scanbound.annotations import Annotation, Ellipse, Origin, Polygon, Rect
from scanbound.ml import RenderSpec, evaluate, export_yolo, review_stats
from scanbound.ml.clips import export_clips
from scanbound.ml.evaluate import average_precision, geometry_iou
from scanbound.project import Project


@pytest.fixture()
def project(tmp_path):
    for k in range(4):
        arr = np.zeros((64, 96), np.uint8)
        arr[10:20, 10 + k:30 + k] = 200
        PILImage.fromarray(arr).save(tmp_path / f"img{k}.png")
    p = Project.open_folder(str(tmp_path))
    for k, iid in enumerate(p.image_ids()):
        iid = p.entry(iid).id
        p.annotations.add(Annotation(iid, "feature", Rect(10 + k, 10, 20, 10), label="bar"))
        p.annotations.add(Annotation(iid, "feature", Ellipse(50, 30, 10, 10), label="dot"))
        p.annotations.add(Annotation(iid, "feature", Rect(70, 40, 5, 5), label="bar", status="proposed"))
    return p


def test_export_only_ground_truth_with_manifest(project, tmp_path):
    out = tmp_path / "ds"
    rep = export_yolo(project, str(out), task="segment", val_ratio=0.25, seed=1, render=RenderSpec("display_gray"))
    assert rep.classes == ["bar", "dot"]
    assert rep.counts == {"train": 3, "val": 1}
    assert rep.instances == {"bar": 4, "dot": 4}                 # the proposals are not exported
    man = json.loads((out / "manifest.json").read_text())
    assert man["render"]["mode"] == "display_gray" and len(man["images"]) == 4
    first = man["images"][0]
    assert first["sha256"] and first["chain"]["steps"] == [] and len(first["annotations"]) == 2
    label = out / ("labels/" + first["image"].split("/", 1)[1].rsplit(".", 1)[0] + ".txt")
    lines = label.read_text().strip().splitlines()
    assert len(lines) == 2 and all(len(l.split()) >= 7 for l in lines)   # polygons
    vals = np.array([float(v) for l in lines for v in l.split()[1:]])
    assert vals.min() >= 0 and vals.max() <= 1
    again = export_yolo(project, str(tmp_path / "ds2"), task="detect", val_ratio=0.25, seed=1)
    assert [e["split"] for e in json.loads((tmp_path / "ds2" / "manifest.json").read_text())["images"]] == \
        [e["split"] for e in man["images"]]                       # the split depends only on the seed


def test_export_without_ground_truth_fails_clearly(tmp_path):
    PILImage.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "x.png")
    p = Project.open_folder(str(tmp_path))
    with pytest.raises(ValueError, match="ground-truth"):
        export_yolo(p, str(tmp_path / "ds"))


def _pred(iid, geom, label, conf):
    return Annotation(iid, "feature", geom, label=label, status="proposed", origin=Origin("model", "m", confidence=conf))


def test_evaluation_counts_and_ap():
    gt = [Annotation("a", "feature", Rect(0, 0, 10, 10), label="x"), Annotation("a", "feature", Rect(20, 0, 10, 10), label="x"),
          Annotation("b", "feature", Rect(0, 0, 10, 10), label="y")]
    pred = [_pred("a", Rect(1, 0, 10, 10), "x", 0.9),             # TP
            _pred("a", Rect(1, 1, 10, 10), "x", 0.8),             # duplicate -> FP
            _pred("a", Rect(50, 50, 5, 5), "x", 0.3),             # FP
            _pred("b", Rect(0, 0, 10, 10), "x", 0.7)]             # wrong class -> FP, y missed
    ev = evaluate(gt, pred, 0.5)
    x, y = ev.per_class["x"], ev.per_class["y"]
    assert (x.tp, x.fp, x.fn) == (1, 3, 1)
    assert (y.tp, y.fp, y.fn) == (0, 0, 1)
    assert x.precision == pytest.approx(0.25) and x.recall == pytest.approx(0.5)
    assert 0 < x.ap < 1 and y.ap == 0.0
    agnostic = evaluate(gt, pred, 0.5, class_aware=False)
    assert agnostic.overall.tp == 2
    assert "mAP" in ev.table()
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_mask_iou_differs_from_box_iou():
    tri = Polygon([[0, 0], [10, 0], [0, 10]])
    box = Rect(0, 0, 10, 10)
    assert geometry_iou(tri, box, "box") == pytest.approx(1.0)
    assert geometry_iou(tri, box, "mask") == pytest.approx(0.5, abs=0.05)


def test_review_statistics(project):
    s = project.annotations
    props = s.query(lambda a: a.status == "proposed")
    for a in props:
        s.update(a.id, origin=Origin("detector", "adapt", run="r1"))
    a, b, c = props[:3]
    s.accept(a.id)
    s.update(b.id, geometry=b.geometry.translated(1, 0))
    s.accept(b.id)
    s.reject(c.id)
    st = review_stats(s, "r1")
    assert st["accepted"] == 1 and st["accepted_edited"] == 1 and st["rejected"] == 1 and st["pending"] == 1
    assert st["acceptance_rate"] == pytest.approx(2 / 3)


def test_analysis_on_anisotropic_pixels(tmp_path):
    import shutil
    shutil.copy(os.path.join(os.path.dirname(__file__), "data", "aniso_0020.sxm"), tmp_path / "s.sxm")
    p = Project.open_folder(str(tmp_path))
    iid = p.entry("s.sxm::").id
    img = p.raw_image(iid)
    p.annotations.add(Annotation(iid, "feature", Rect(10, 10, 20, 20), label="m"))
    p.annotations.add(Annotation(iid, "feature", Rect(40, 10, 4, 4), label="m", status="proposed"))   # not analysed
    ctx = analysis_context(p, [iid])
    assert len(ctx.objects) == 1
    dx, dy = img.pixel_size
    assert ctx.objects["area"].iloc[0] == pytest.approx(20 * dx * 20 * dy * 1e18, rel=1e-3)


def test_feature_clips(project, tmp_path):
    n = export_clips(project, str(tmp_path / "clips"), per_group=2)
    assert n == 4
    assert sorted(os.listdir(tmp_path / "clips")) == ["bar", "dot"]
