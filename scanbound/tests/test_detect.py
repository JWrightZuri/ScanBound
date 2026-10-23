from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from scanbound import DetectionConfig, Detector, LabelingConfig, NoMaskError, find_ribbons
from scanbound.cli import main
from scanbound.config import PipelineConfig


def _config():
    cfg = DetectionConfig.default()
    cfg.pipeline = PipelineConfig.from_tuples([("gaussian", True, {"sigma": 1.0}),
                                               ("background", True, {"sigma": 40.0}),
                                               ("threshold", True, {"method": "Otsu", "offset": 0.05}),
                                               ("size_filter", True, {"min_size": 20})])
    return cfg


def test_detects_blobs(blobs):
    r = Detector(_config()).detect(blobs, px_scale=0.5)
    assert r.n_objects == 6 and not r.errors
    assert set(r.labels.values()) == {None}               # single mode without a label: no ROIs
    assert r.labeled_objects() == []
    areas = sorted(r.features["area"])
    assert areas[-1] > areas[0] * 4 and r.features.index.tolist() == [o.id for o in r.objects]


def test_rule_labels(blobs):
    cfg = _config()
    cfg.labeling = LabelingConfig.from_dict({"mode": "rules", "fallback": "small", "rules": [
        {"label": "large", "conditions": [{"feature": "eq_diameter", "min": 8.0}]}]})   # nm, px_scale 0.5
    r = Detector(cfg).detect(blobs, px_scale=0.5)
    assert sorted(r.labels.values()) == ["large", "large", "small", "small", "small", "small"]
    rows = r.annotation_rows("x.png")
    assert {row[7] for row in rows} == {"large", "small"} and all(row[6] == "x.png" for row in rows)


def test_pipeline_without_threshold_is_rejected():
    cfg = DetectionConfig.default()
    cfg.pipeline = PipelineConfig.from_tuples([("gaussian", True, {})])
    with pytest.raises(NoMaskError):
        Detector(cfg)


def test_ribbons(ribbon_mask):
    split = find_ribbons(ribbon_mask, px_scale=0.2, min_length_px=10)
    lengths = sorted(r.length_px for r in split)
    assert len(split) == 4                                  # straight + 3 arms of the T
    assert lengths[-1] == pytest.approx(100, abs=4)         # the straight one
    true_width = int(ribbon_mask[:60, 70].sum())           # cv2 draws thickness 5 as 7 px
    assert all(r.width_px == pytest.approx(true_width, abs=0.5) for r in split)
    assert split[0].length_nm == pytest.approx(split[0].length_px * 0.2)
    whole = find_ribbons(ribbon_mask, min_length_px=10, split_at_junctions=False)
    assert len(whole) == 2


def test_nanoribbon_preset_on_drawn_ribbons():
    rng = np.random.default_rng(1)
    img = np.full((300, 300), 0.3, np.float32)
    for (a, b) in [((20, 40), (200, 60)), ((50, 150), (250, 260)), ((280, 20), (230, 200))]:
        cv2.line(img, a, b, 1.0, 6)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    r = Detector(DetectionConfig.nanoribbons()).detect(img, px_scale=0.1)
    long = [x for x in r.ribbons if x.length_px > 100]
    assert len(long) == 3
    assert r.summary()["ribbons"] >= 3 and "ribbon_mean_length_nm" in r.summary()


@pytest.fixture
def image_files(tmp_path, blobs):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), (np.clip(np.roll(blobs, 20 * i, axis=1), 0, 1) * 255).astype(np.uint8))
        paths.append(str(p))
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not a png")
    return paths, str(bad)


def test_detect_many_parallel_matches_serial(image_files):
    paths, bad = image_files
    det = Detector(_config())
    serial = {i.path: i for i in det.detect_many(paths + [bad])}
    parallel = {i.path: i for i in det.detect_many(paths + [bad], workers=2)}
    assert not serial[bad].ok and not parallel[bad].ok
    for p in paths:
        assert serial[p].result.n_objects == parallel[p].result.n_objects >= 6
        assert np.array_equal(serial[p].result.mask, parallel[p].result.mask)


def test_cli(image_files, tmp_path, saved_params_path):
    paths, _ = image_files
    out = tmp_path / "out"
    assert main(["detect", "-c", saved_params_path, "-o", str(out), "--overlays", "--label", "blob", *paths]) == 0
    ann = pd.read_csv(out / "annotations.csv")
    assert set(ann["label"]) == {"blob"} and set(ann["file"]) == {os.path.basename(p) for p in paths}
    assert (out / "features.csv").exists() and (out / "img_0_overlay.png").exists()
    assert main(["check-config", saved_params_path]) == 0


def test_core_does_not_import_qt():
    code = "import sys, scanbound, scanbound.cli, scanbound.export; " \
           "bad = [m for m in sys.modules if m.startswith(('PyQt', 'matplotlib', 'tkinter'))]; print(bad); " \
           "sys.exit(1 if bad else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
