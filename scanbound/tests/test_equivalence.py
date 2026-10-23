"""The new package reproduces the legacy core exactly on real images."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skimage.io import imread

from scanbound import ADAPT_STEPS, DetectionConfig, Detector, Pipeline, RoiSettings, extract_objects
from scanbound.labeling import LabelingConfig, assign_labels
from scanbound.measure import measure_objects
from scanbound.steps import SPM_STEPS


def _legacy_steps(legacy, cfg):
    reg = legacy["image_adapt_dialog"].STEP_REGISTRY
    return [(reg[s.type], dict(s.params), s.enabled) for s in cfg.pipeline.steps]


@pytest.fixture(scope="module")
def images(sample_paths):
    from scanbound._image import to_float_gray
    return [to_float_gray(imread(p, as_gray=True)) for p in sample_paths[:8]]


def test_registry_matches_legacy(legacy):
    old = legacy["image_adapt_dialog"].STEP_REGISTRY
    for key, spec in old.items():
        new = ADAPT_STEPS[key]
        assert new.needs == spec.needs
        assert [(p.key, p.kind, p.default, p.minimum, p.maximum, tuple(p.choices)) for p in new.params] == \
               [(p.key, p.kind, p.default, p.minimum, p.maximum, tuple(p.choices)) for p in spec.params]
    old_spm = legacy["spm_processing"].SPM_STEPS
    assert list(SPM_STEPS) == list(old_spm)


@pytest.mark.parametrize("preset", ["saved", "nanoribbons", "adaptive"])
def test_pipeline_masks_identical(legacy, images, saved_params_path, preset):
    if preset == "saved":
        cfg = DetectionConfig.load(saved_params_path)
    elif preset == "nanoribbons":
        cfg = DetectionConfig.nanoribbons()
        cfg.pipeline.steps = [s for s in cfg.pipeline.steps if s.type != "rescale"]   # not in legacy
    else:
        cfg = DetectionConfig.default()
        cfg.pipeline.steps[2].params.update(method="Adaptive (Mean)", block=41, offset=0.02)
        cfg.pipeline.steps.insert(0, DetectionConfig.default().pipeline.steps[0].__class__(
            "clahe", True, ADAPT_STEPS["clahe"].defaults()))
    pipe = Pipeline.from_config(cfg.pipeline)
    run_old = legacy["image_adapt_dialog"].run_pipeline
    for img in images:
        old, old_err = run_old(img, _legacy_steps(legacy, cfg))
        new = pipe.run(img)
        assert not old_err and new.ok
        assert new.image.dtype == old.dtype == np.bool_
        assert np.array_equal(new.image, old)


@pytest.mark.parametrize("roi_type", ["RectROI", "PolygonROI", "LineSegmentROI", "PointROI"])
def test_objects_and_features_identical(legacy, images, saved_params_path, roi_type):
    cfg = DetectionConfig.load(saved_params_path)
    pipe = Pipeline.from_config(cfg.pipeline)
    old_roi, old_ol = legacy["roi_extraction"], legacy["object_labeling"]
    settings = RoiSettings(roi_type=roi_type, rect_padding=1.0)
    for img in images[:4]:
        mask = pipe.run(img).image
        new_objs = extract_objects(mask, settings)
        old_objs = old_roi.extract_objects(mask, old_roi.RoiSettings(**settings.to_dict()))
        assert [o.id for o in new_objs] == [o.id for o in old_objs]
        for a, b in zip(new_objs, old_objs):
            assert np.array_equal(a.points, b.points) and a.bbox == b.bbox and a.rect == b.rect
        new_f = measure_objects(mask, new_objs, 0.25, img)
        old_f = old_ol.measure_objects(mask, old_objs, 0.25, img)
        pd.testing.assert_frame_equal(new_f, old_f)


def test_labels_identical(legacy, images, saved_params_path):
    cfg = DetectionConfig.load(saved_params_path)
    det = Detector(cfg)
    old_ol = legacy["object_labeling"]
    configs = [
        {"mode": "rules", "fallback": "small", "rules": [
            {"label": "big", "conditions": [{"feature": "area", "min": 60, "max": None}]},
            {"label": "long", "conditions": [{"feature": "aspect_ratio", "min": 1.5, "max": None}]}]},
        {"mode": "bins", "bins": {"feature": "eq_diameter", "thresholds": [8.0, 10.0], "labels": ["s", "m", "l"]}},
        {"mode": "clusters", "clusters": {"features": ["area", "aspect_ratio"], "k": 3, "labels": ["a", "b", "c"]}},
    ]
    for img in images[:3]:
        r = det.detect(img)
        for c in configs:
            full = {**old_ol.default_config("x"), **c}
            assert assign_labels(r.features, LabelingConfig.from_dict(full)) == old_ol.assign_labels(r.features, full)


def test_spm_steps_identical(legacy, images):
    old = legacy["spm_processing"].SPM_STEPS
    z = images[0].astype(float) * 3e-9
    for key, spec in SPM_STEPS.items():
        params = spec.defaults()
        if key == "fourier":
            params["shapes"] = [["ellipse", 0.1, 0.0, 0.02, 0.02]]
        assert np.allclose(spec.func(z, **params), old[key].func(z, **params), equal_nan=True), key
