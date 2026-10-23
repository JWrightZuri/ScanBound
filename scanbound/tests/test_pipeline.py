from __future__ import annotations

import numpy as np
import pytest

from scanbound import ADAPT_STEPS, Param, Pipeline, PipelineCache, PipelineError, StepRegistry, StepSpec, run_pipeline


def test_binary_step_before_threshold_is_reported(blobs):
    res = Pipeline([("size_filter", True, {}), ("threshold", True, {})]).run(blobs)
    assert res.is_mask and [e.index for e in res.errors] == [0]
    with pytest.raises(PipelineError, match="step 0 \\(size_filter\\)"):
        Pipeline([("size_filter", True, {}), ("threshold", True, {})]).run(blobs, strict=True)


def test_failing_step_is_isolated(blobs):
    def boom(img, x):
        raise RuntimeError("bad")
    reg = ADAPT_STEPS.extended("test", StepSpec("boom", "Boom", "gray", boom, (Param("x", "X", "int", 1, 0, 5),)))
    res = Pipeline([("boom", True, {}), ("threshold", True, {})], reg).run(blobs)
    assert res.errors[0].message == "RuntimeError: bad" and res.is_mask
    with pytest.raises(PipelineError):
        Pipeline([("boom", True, {})], reg).run(blobs, strict=True)


def test_cache_skips_unchanged_steps(blobs):
    calls = []

    def counted(img, sigma):
        calls.append(sigma)
        return img

    reg = ADAPT_STEPS.extended("test", StepSpec("count", "Count", "gray", counted,
                                                (Param("sigma", "S", "float", 1.0, 0, 10),)))
    cache = PipelineCache()
    Pipeline([("count", True, {"sigma": 1}), ("threshold", True, {})], reg).run(blobs, cache=cache)
    Pipeline([("count", True, {"sigma": 1}), ("threshold", True, {"offset": 0.1})], reg).run(blobs, cache=cache)
    assert calls == [1.0]
    Pipeline([("count", True, {"sigma": 2}), ("threshold", True, {})], reg).run(blobs, cache=cache)
    assert calls == [1.0, 2.0]
    Pipeline([("count", True, {"sigma": 2})], reg).run(blobs.copy(), cache=cache)   # new image: recompute
    assert calls == [1.0, 2.0, 2.0]


def test_legacy_run_pipeline_signature(blobs):
    steps = [(ADAPT_STEPS["gaussian"], {"sigma": 1.0}, True), (ADAPT_STEPS["threshold"],
             ADAPT_STEPS["threshold"].defaults(), True), (ADAPT_STEPS["morph"], ADAPT_STEPS["morph"].defaults(), False)]
    cache: list = []
    img, errors = run_pipeline(blobs, steps, cache)
    assert img.dtype == np.bool_ and errors == {} and len(cache) == 3
    img2, _ = run_pipeline(blobs, steps, cache)
    assert np.array_equal(img, img2)


def test_registry_rejects_duplicates():
    with pytest.raises(ValueError):
        StepRegistry("x", [ADAPT_STEPS["gaussian"], ADAPT_STEPS["gaussian"]])
