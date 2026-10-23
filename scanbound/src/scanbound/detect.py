"""
The detection entry point: image (or file) in, objects / ROIs / measurements / labels out.

    from scanbound import Detector, DetectionConfig

    det = Detector(DetectionConfig.load("image_adapt_params.json"))
    result = det.detect(image, px_scale=0.1)           # numpy image, nm per pixel
    result.objects, result.features, result.labels, result.ribbons

    for item in det.detect_many(paths, workers=4):      # files, in parallel processes
        if item.ok:
            item.result.features.to_csv(...)

No Qt, no matplotlib, no global state: the same Detector serves the GUI, scripts and the CLI.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ._image import FloatImage, Mask, to_float_gray
from .config import DetectionConfig
from .errors import ScanBoundError
from .labeling import assign_labels
from .measure import measure_objects
from .pipeline import Pipeline, PipelineCache, StepFailure
from .ribbons import Ribbon, find_ribbons, ribbons_table
from .roi import SegObject, extract_objects, mask_components, objects_to_rows
from .steps.adapt import ADAPT_STEPS
from .steps.registry import StepRegistry

log = logging.getLogger(__name__)
PathLike = Union[str, "os.PathLike[str]"]


class NoMaskError(ScanBoundError):
    """The pipeline ended on a gray image, so there is nothing to split into objects."""


@dataclass
class Detection:
    image: FloatImage                               # input as float gray in [0, 1]
    mask: Mask                                      # pipeline output
    objects: list[SegObject]
    labels: dict[int, Optional[str]]                # object id -> label (None = skipped)
    features: Optional[pd.DataFrame] = None         # index = object id, see scanbound.measure.FEATURES
    ribbons: Optional[list[Ribbon]] = None
    px_scale: Optional[float] = None                # nm per pixel
    errors: list[StepFailure] = field(default_factory=list)
    path: Optional[str] = None
    channel: Optional[str] = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def labeled_objects(self) -> list[tuple[SegObject, str]]:
        """``(object, label)`` for every object that gets an ROI (label not None)."""
        out = []
        for o in self.objects:
            label = self.labels.get(o.id)
            if label is not None:
                out.append((o, label))
        return out

    def annotation_rows(self, filename: Optional[str] = None) -> list[list]:
        """Rows in the ``annotations_df`` layout (objectName, type, x1, y1, dx, dy, file, label)."""
        name = filename or (os.path.basename(self.path) if self.path else "")
        rows: list[list] = []
        for obj, label in self.labeled_objects():
            rows.extend(objects_to_rows([obj], name, label))
        return rows

    def ribbons_table(self) -> pd.DataFrame:
        return ribbons_table(self.ribbons or [])

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for _, label in self.labeled_objects():
            counts[label] = counts.get(label, 0) + 1
        out: dict[str, Any] = {"file": os.path.basename(self.path) if self.path else None,
                               "channel": self.channel, "px_scale_nm": self.px_scale,
                               "objects": self.n_objects, "labeled": sum(counts.values()), "labels": counts,
                               "mask_fraction": float(self.mask.mean()) if self.mask.size else 0.0,
                               "errors": [f"step {e.index} ({e.step}): {e.message}" for e in self.errors],
                               "seconds": round(sum(self.timings.values()), 4)}
        if self.ribbons is not None:
            lengths = np.array([r.length_px for r in self.ribbons], float)
            out["ribbons"] = len(self.ribbons)
            if lengths.size:
                scale = self.px_scale or 1.0
                unit = "nm" if self.px_scale else "px"
                out[f"ribbon_mean_length_{unit}"] = float(lengths.mean() * scale)
                out[f"ribbon_median_length_{unit}"] = float(np.median(lengths) * scale)
        return out

    def overlay(self, object_colors: Optional[dict[int, Optional[str]]] = None) -> np.ndarray:
        """RGB uint8 overlay of the objects and their ROIs (ribbon centre lines if analysed)."""
        from .render import render_objects, render_ribbons
        if self.ribbons is not None:
            return render_ribbons(self.image, self.mask, self.ribbons)
        return render_objects(self.image, self.mask, self.objects, object_colors)


@dataclass
class BatchItem:
    path: str
    result: Optional[Detection] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Detector:
    """
    Runs a :class:`DetectionConfig` on images.

    ``strict``: a failing pipeline step raises :class:`scanbound.errors.PipelineError` instead of
    being skipped and reported in ``Detection.errors``.
    ``measure``: always compute the feature table (otherwise only when labelling needs it).
    """

    def __init__(self, config: Optional[DetectionConfig] = None, registry: StepRegistry = ADAPT_STEPS,
                 strict: bool = False, measure: bool = True):
        self.config = config or DetectionConfig()
        self.registry = registry
        self.strict = strict
        self.measure = measure
        self.pipeline = Pipeline.from_config(self.config.pipeline, registry)
        if not self.pipeline.produces_mask:
            raise NoMaskError("the pipeline has no enabled Threshold step, so it cannot produce a binary mask")
        self._cache: Optional[PipelineCache] = None

    def __repr__(self) -> str:
        return f"Detector({self.pipeline!r}, roi={self.config.roi.roi_type}, labeling={self.config.labeling.mode})"

    def enable_cache(self) -> PipelineCache:
        """Reuse unchanged pipeline steps between calls on the same image array (interactive use)."""
        self._cache = PipelineCache()
        return self._cache

    def detect(self, image: np.ndarray, px_scale: Optional[float] = None, *,
               path: Optional[str] = None, channel: Optional[str] = None,
               valid: Optional[np.ndarray] = None) -> Detection:
        """``valid``: bool mask of pixels that hold data; objects are never found outside it."""
        cfg = self.config
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = now - t
            t = now

        gray = to_float_gray(image)
        res = self.pipeline.run(gray, strict=self.strict, cache=self._cache)
        lap("pipeline")
        if not res.is_mask:
            raise NoMaskError("the pipeline output is not a binary mask (is the Threshold step failing?): "
                              + "; ".join(e.message for e in res.errors))
        mask = res.image.astype(bool, copy=False)
        if valid is not None and not np.all(valid):
            mask = mask & np.asarray(valid, bool)
        components = mask_components(mask)
        objects = extract_objects(mask, cfg.roi, components=components)
        lap("objects")

        features = None
        if self.measure or cfg.labeling.needs_features:
            features = measure_objects(mask, objects, px_scale, intensity=gray, components=components)
            lap("measure")
        if cfg.labeling.needs_features:
            assert features is not None
            labels = assign_labels(features, cfg.labeling)
        else:
            labels = {o.id: cfg.labeling.single_label for o in objects}
        lap("labels")

        ribbons = None
        if cfg.ribbons.enabled:
            ribbons = find_ribbons(mask, px_scale, cfg.ribbons.min_length_px, cfg.ribbons.split_at_junctions,
                                   components=components)
            lap("ribbons")
        if res.errors:
            for e in res.errors:
                log.warning("%s: step %d (%s) failed: %s", path or "image", e.index, e.step, e.message)
        return Detection(gray, mask, objects, labels, features, ribbons, px_scale, res.errors,
                         path, channel, timings)

    def detect_file(self, path: PathLike, channel: Optional[str] = None,
                    px_scale: Optional[float] = None) -> Detection:
        from .io import load_image
        img = load_image(path, channel, px_scale)
        return self.detect(img.data, img.px_scale, path=img.path, channel=img.channel)

    def detect_many(self, paths: Iterable[PathLike], workers: Optional[int] = None,
                    channel: Optional[str] = None, px_scale: Optional[float] = None) -> Iterator[BatchItem]:
        """
        Detect in many files.  ``workers`` > 1 uses that many processes (results arrive in
        completion order); ``None`` or 1 runs in this process, in order.  A file that fails
        yields a :class:`BatchItem` with ``error`` set instead of stopping the batch.
        """
        files = [os.fspath(p) for p in paths]
        if not workers or workers <= 1 or len(files) <= 1:
            for p in files:
                yield _detect_one(self, p, channel, px_scale)
            return
        if self.registry is not ADAPT_STEPS:
            raise ScanBoundError("parallel detection only supports the built-in step registry")
        cfg = self.config.to_dict()
        # spawn, as on Windows: the same behaviour everywhere and no fork of a threaded process
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_worker, cfg, self.strict, self.measure, p, channel, px_scale) for p in files]
            for fut in as_completed(futures):
                yield fut.result()


def _detect_one(det: Detector, path: str, channel: Optional[str], px_scale: Optional[float]) -> BatchItem:
    try:
        return BatchItem(path, det.detect_file(path, channel, px_scale))
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
        log.error("%s: %s: %s", path, type(exc).__name__, exc)
        return BatchItem(path, error=f"{type(exc).__name__}: {exc}")


def _worker(cfg: dict, strict: bool, measure: bool, path: str, channel: Optional[str],
            px_scale: Optional[float]) -> BatchItem:
    det = Detector(DetectionConfig.from_dict(cfg), strict=strict, measure=measure)
    return _detect_one(det, path, channel, px_scale)
