"""Reproducible processing and the project file."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from scanbound.annotations import Annotation, Rect
from scanbound.io import open_data
from scanbound.model import Image
from scanbound.processing import ProcessingChain
from scanbound.project import DEFAULT_NAME, Project, ProjectError

DATA = os.path.join(os.path.dirname(__file__), "data")
ANISO = os.path.join(DATA, "aniso_0020.sxm")


def tilted(nan: bool = True) -> Image:
    yy, xx = np.mgrid[0:64, 0:80]
    data = 0.01 * xx + 0.02 * yy + np.sin(xx / 3.0) * 0.1
    if nan:
        data[50:, 60:] = np.nan
    return Image(data, quantity="Z", unit="m")


def test_chain_records_every_step_and_keeps_nan():
    img = tilted()
    chain = ProcessingChain.from_tuples([("plane", True, {}), ("gaussian", True, {"sigma": 1.0}),
                                         ("median", False, {})])
    out = chain.apply(img)
    ops = [r.op for r in out.history]
    assert ops[-3:] == ["spm.plane", "spm.gaussian", "nan_policy"]
    assert np.isnan(out.data[55, 70]) and np.isfinite(out.data[10, 10])
    assert abs(np.nanmean(out.data)) < 0.05                       # the tilt is gone
    again = ProcessingChain.from_history(out).apply(img)
    np.testing.assert_allclose(again.data, out.data, equal_nan=True)


def test_chain_round_trips_through_json_and_rejects_bad_input():
    chain = ProcessingChain.from_tuples([("plane", True, {}), ("align_rows", True, {})])
    back = ProcessingChain.from_dict(json.loads(json.dumps(chain.to_dict())))
    assert back.to_dict() == chain.to_dict()
    with pytest.raises(Exception):
        ProcessingChain.from_dict({"format": "something else"})
    with pytest.raises(Exception):
        ProcessingChain.from_dict({"steps": "plane"})


def test_invalid_parameters_fail_early_with_the_step_name():
    from scanbound.errors import ParamError
    with pytest.raises(ParamError, match="gaussian"):
        chain = ProcessingChain.from_tuples([("plane", True, {}), ("gaussian", True, {"sigma": "not a number"})])
        chain.apply(tilted(False), strict=False)


def test_regenerate_from_the_raw_file():
    img = open_data(ANISO).image()
    out = ProcessingChain.from_tuples([("plane", True, {}), ("align_rows", True, {})]).apply(img)
    again = ProcessingChain.regenerate(out)
    np.testing.assert_allclose(again.data, out.data, equal_nan=True)


def test_project_save_load_and_rebase(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(ANISO, data / "scan.sxm")
    p = Project.open_folder(str(data))
    iid = p.entry("scan.sxm::").id
    p.set_chain(iid, ProcessingChain.from_tuples([("plane", True, {})]))
    p.set_calibration(iid, 2e-10, 3e-10, 0.05)
    p.labels.add("molecule")
    a = Annotation(iid, "feature", Rect(1, 2, 3, 4), label="molecule")
    p.annotations.add(a)
    assert p.dirty
    path = p.save()
    assert os.path.basename(path) == DEFAULT_NAME and not p.dirty
    q = Project.load(path)
    assert q.annotations[a.id].geometry == a.geometry
    assert q.entry(iid).chain.to_dict() == p.entry(iid).chain.to_dict()
    assert q.raw_image(iid).pixel_size == pytest.approx((2e-10, 3e-10))
    assert q.raw_image(iid).x.calibration.source == "manual"
    # saved somewhere else: file references follow
    other = tmp_path / "elsewhere"
    other.mkdir()
    q.save(str(other / "copy.json"))
    r = Project.load(str(other / "copy.json"))
    rid = next(iter(r.images))
    assert r.processed_image(rid).shape == (50, 144)
    assert len(r.annotations.for_image(rid)) == 1


def test_project_notices_changed_and_missing_files(tmp_path):
    shutil.copy(ANISO, tmp_path / "scan.sxm")
    p = Project.open_folder(str(tmp_path))
    assert p.changed_files() == []
    with open(tmp_path / "scan.sxm", "ab") as fh:
        fh.write(b"x")
    assert p.changed_files() == ["scan.sxm"]
    os.remove(tmp_path / "scan.sxm")
    assert p.changed_files() == ["scan.sxm"]


def test_bad_project_files(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("{not json")
    with pytest.raises(ProjectError):
        Project.load(str(bad))
    bad.write_text(json.dumps({"format": "scanbound_project", "version": 999}))
    with pytest.raises(ProjectError):
        Project.load(str(bad))
