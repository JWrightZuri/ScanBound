"""
Command line interface.

    scanbound detect -c image_adapt_params.json -o results/ scans/*.sxm --workers 4 --overlays
    scanbound detect --preset nanoribbons --px-scale 0.1 -o results/ image.png
    scanbound steps                         # list the available pipeline steps
    scanbound check-config my_params.json   # validate a saved pipeline

(``python -m scanbound ...`` works the same.)
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from typing import Optional, Sequence

from . import __version__
from .config import DetectionConfig
from .detect import Detector
from .errors import ScanBoundError
from .steps import ADAPT_STEPS


def _expand(patterns: Sequence[str]) -> list[str]:
    from .io import find_scans, supported_extensions
    out: list[str] = []
    for p in patterns:
        if os.path.isdir(p):
            out.extend(find_scans(p))
            continue
        hits = sorted(glob.glob(p)) or [p]
        exts = tuple(supported_extensions())
        out.extend(h for h in hits if os.path.isdir(h) is False and (h.lower().endswith(exts) or h == p))
    return list(dict.fromkeys(out))                          # de-duplicate, keep order


def _cmd_detect(a: argparse.Namespace) -> int:
    if a.config:
        cfg = DetectionConfig.load(a.config, strict=a.strict)
    else:
        cfg = DetectionConfig.nanoribbons() if a.preset == "nanoribbons" else DetectionConfig.default()
    for w in cfg.warnings:
        logging.warning("config: %s", w)
    if a.roi_type:
        cfg.roi.roi_type = a.roi_type
    if a.ribbons:
        cfg.ribbons.enabled = True
    if cfg.labeling.mode == "single" and cfg.labeling.single_label is None:
        cfg.labeling.single_label = a.label

    files = _expand(a.inputs)
    if not files:
        logging.error("no input files found")
        return 2
    det = Detector(cfg, strict=a.strict)
    logging.info("%r on %d file(s)", det, len(files))
    t0 = time.perf_counter()
    results, failed = [], 0
    for item in det.detect_many(files, workers=a.workers, channel=a.channel, px_scale=a.px_scale):
        if item.ok and item.result is not None:
            r = item.result
            results.append(r)
            extra = f", {len(r.ribbons)} ribbons" if r.ribbons is not None else ""
            logging.info("%s: %d objects%s", os.path.basename(item.path), r.n_objects, extra)
        else:
            failed += 1
            logging.error("%s: %s", item.path, item.error)
    results.sort(key=lambda r: r.path or "")
    from .export import write_results
    written = write_results(results, a.output, overlays=a.overlays)
    logging.info("done in %.2f s: %d ok, %d failed; wrote %d file(s) to %s",
                 time.perf_counter() - t0, len(results), failed, len(written), a.output)
    return 1 if failed else 0


def _cmd_steps(_: argparse.Namespace) -> int:
    for spec in ADAPT_STEPS.values():
        print(f"{spec.key:<12} [{spec.needs:<6}] {spec.title}: {spec.description}")
        for p in spec.params:
            rng = f"one of {', '.join(p.choices)}" if p.kind == "choice" else (
                "true/false" if p.kind == "bool" else f"{p.minimum}..{p.maximum}")
            print(f"    {p.key:<11} {p.kind:<6} default {p.default!r:<10} {rng}")
    return 0


def _cmd_check(a: argparse.Namespace) -> int:
    cfg = DetectionConfig.load(a.config)
    Detector(cfg)
    for w in cfg.warnings:
        print(f"warning: {w}")
    steps = ", ".join(s.type + ("" if s.enabled else " (off)") for s in cfg.pipeline.steps)
    print(f"ok: {steps}; ROI {cfg.roi.roi_type}; labelling {cfg.labeling.mode}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="scanbound", description="ScanBound detection core")
    ap.add_argument("--version", action="version", version=f"scanbound {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="detect objects in images and write CSV / JSON results")
    d.add_argument("inputs", nargs="+", help="files, folders or glob patterns")
    d.add_argument("-o", "--output", default="scanbound_results", help="output folder")
    src = d.add_mutually_exclusive_group()
    src.add_argument("-c", "--config", help="pipeline JSON saved by the image adaptation dialog")
    src.add_argument("--preset", choices=("default", "nanoribbons"), default="default")
    d.add_argument("--px-scale", type=float, help="nm per pixel (overrides the file calibration)")
    d.add_argument("--channel", help="channel name (default: topography / first channel)")
    d.add_argument("--roi-type", choices=("RectROI", "PolygonROI", "LineSegmentROI", "PointROI"))
    d.add_argument("--label", default="object", help="label for single-label configs without one")
    d.add_argument("--ribbons", action="store_true", help="also measure ribbon centre lines")
    d.add_argument("--workers", type=int, default=1, help="parallel processes")
    d.add_argument("--overlays", action="store_true", help="save an overlay PNG per image")
    d.add_argument("--strict", action="store_true", help="fail on config warnings and step errors")
    d.set_defaults(func=_cmd_detect)

    s = sub.add_parser("steps", help="list the pipeline steps and their parameters")
    s.set_defaults(func=_cmd_steps)

    c = sub.add_parser("check-config", help="validate a pipeline JSON file")
    c.add_argument("config")
    c.set_defaults(func=_cmd_check)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        return int(a.func(a))
    except ScanBoundError as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
