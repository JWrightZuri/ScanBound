from __future__ import annotations

import json

import pytest

from scanbound import ConfigError, DetectionConfig, ParamError, UnknownStepError
from scanbound.labeling import LabelingConfig


def test_saved_file_round_trip(saved_params_path, tmp_path):
    cfg = DetectionConfig.load(saved_params_path, strict=True)
    assert [s.type for s in cfg.pipeline.steps] == ["gaussian", "background", "threshold", "morph", "size_filter"]
    assert cfg.pipeline.steps[2].params["offset"] == 0.06
    assert cfg.roi.roi_type == "PolygonROI" and cfg.show_objects
    out = tmp_path / "p.json"
    cfg.save(out)
    again = DetectionConfig.load(out)
    assert again == cfg
    original = json.load(open(saved_params_path))
    written = json.load(open(out))
    assert written["steps"] == original["steps"] and written["roi"] == original["roi"]


def _cfg(steps, **extra):
    return {"format": "image_adapt_pipeline", "version": 1, "steps": steps, **extra}


def test_defaults_filled_in():
    cfg = DetectionConfig.from_dict(_cfg([{"type": "threshold"}]))
    assert cfg.pipeline.steps[0].params == {"method": "Otsu", "features": "Bright", "value": 0.5,
                                            "block": 31, "offset": 0.0}
    assert cfg.pipeline.steps[0].enabled


def test_repairable_problems_warn():
    cfg = DetectionConfig.from_dict(_cfg([
        {"type": "gaussian", "params": {"sigma": 99}},
        {"type": "threshold", "params": {"method": "Fancy", "block": 30, "bogus": 1}},
    ], roi={"roi_type": "RectROI", "colour": "red"}))
    p0, p1 = cfg.pipeline.steps[0].params, cfg.pipeline.steps[1].params
    assert p0["sigma"] == 10.0 and p1["method"] == "Otsu" and p1["block"] == 31
    assert len(cfg.warnings) == 5
    with pytest.raises(ConfigError):
        DetectionConfig.from_dict(_cfg([{"type": "gaussian", "params": {"sigma": 99}}]), strict=True)


def test_fatal_problems_raise():
    with pytest.raises(UnknownStepError, match="unknown step 'blur'"):
        DetectionConfig.from_dict(_cfg([{"type": "blur"}]))
    with pytest.raises(ParamError):
        DetectionConfig.from_dict(_cfg([{"type": "gaussian", "params": {"sigma": "wide"}}]))
    with pytest.raises(ConfigError):
        DetectionConfig.from_dict({"steps": "nope"})
    with pytest.raises(ConfigError):
        DetectionConfig.from_dict(_cfg([], roi={"roi_type": "Circle"}))
    with pytest.raises(ConfigError):
        DetectionConfig.from_dict({"format": "scanbound_spm_pipeline", "steps": []})
    with pytest.raises(ConfigError, match="unknown feature"):
        LabelingConfig.from_dict({"mode": "bins", "bins": {"feature": "colour"}})


def test_bad_json_names_the_file(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{nope")
    with pytest.raises(ConfigError, match="broken.json"):
        DetectionConfig.load(p)


def test_labeling_round_trip():
    d = {"mode": "rules", "single_label": None, "fallback": None,
         "rules": [{"label": "rod", "conditions": [{"feature": "aspect_ratio", "min": 3.0, "max": None}]}],
         "bins": {"feature": "area", "thresholds": [10.0], "labels": ["a", "b"], "log": True},
         "clusters": {"features": ["area"], "k": 2, "labels": ["x", "y"], "log": False}}
    assert LabelingConfig.from_dict(d).to_dict() == d
