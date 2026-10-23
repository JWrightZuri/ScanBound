"""
Typed configuration of a detection run, loadable from and savable to the JSON files the
adaptation dialog writes (``"format": "image_adapt_pipeline"``, version 1)::

    cfg = DetectionConfig.load("data/user_data/image_adapt_params.json")
    cfg.pipeline.steps[0].params["sigma"] = 0.5
    cfg.roi.roi_type = "LineSegmentROI"
    cfg.save("my_params.json")              # still opens in the dialog

Loading validates every step and parameter against the step registry.  Problems that can be
repaired (an out-of-range value, an unknown parameter) are collected in ``cfg.warnings``;
``strict=True`` turns them into :class:`ConfigError`.  Things that cannot be repaired (an
unknown step, a wrong type) always raise.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

from .errors import ConfigError
from .labeling import LabelingConfig
from .roi import RoiSettings
from .steps.adapt import ADAPT_STEPS, DEFAULT_ADAPT_PIPELINE, NANORIBBON_PIPELINE
from .steps.registry import StepRegistry

PARAM_FILE_FORMAT = "image_adapt_pipeline"
PARAM_FILE_VERSION = 1
PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class StepConfig:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "enabled": self.enabled, "params": dict(self.params)}


@dataclass
class PipelineConfig:
    steps: list[StepConfig] = field(default_factory=list)

    @classmethod
    def from_tuples(cls, steps: list[tuple[str, bool, dict]], registry: StepRegistry = ADAPT_STEPS) -> "PipelineConfig":
        return cls([StepConfig(k, e, registry[k].resolve(p)) for k, e, p in steps])

    @classmethod
    def from_list(cls, entries: Any, registry: StepRegistry = ADAPT_STEPS,
                  warnings: Optional[list[str]] = None) -> "PipelineConfig":
        if not isinstance(entries, list):
            raise ConfigError("not a valid pipeline: 'steps' must be a list")
        steps = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ConfigError(f"steps[{i}] must be an object with 'type', 'enabled' and 'params'")
            spec = registry[str(entry.get("type"))]                  # raises UnknownStepError
            raw = entry.get("params") or {}
            if not isinstance(raw, Mapping):
                raise ConfigError(f"steps[{i}].params must be an object")
            params = spec.resolve(raw, warnings, where=f"steps[{i}] ({spec.key}).")
            steps.append(StepConfig(spec.key, bool(entry.get("enabled", True)), params))
        return cls(steps)

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    def as_tuples(self) -> list[tuple[str, bool, dict]]:
        return [(s.type, s.enabled, dict(s.params)) for s in self.steps]


@dataclass
class RibbonSettings:
    """Options of the nanoribbon (skeleton branch) analysis, see :mod:`scanbound.ribbons`."""

    enabled: bool = False
    min_length_px: float = 3.0       # shorter skeleton branches (spurs) are dropped
    split_at_junctions: bool = True  # False: one centre line per object (longest path)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "RibbonSettings":
        valid = {f.name for f in fields(cls)}
        out = cls(**{k: v for k, v in (d or {}).items() if k in valid})
        if float(out.min_length_px) < 0:
            raise ConfigError("ribbons.min_length_px must be >= 0")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "min_length_px": self.min_length_px,
                "split_at_junctions": self.split_at_junctions}


@dataclass
class DetectionConfig:
    """Everything a :class:`scanbound.Detector` needs besides the image."""

    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig.from_tuples(DEFAULT_ADAPT_PIPELINE))
    roi: RoiSettings = field(default_factory=RoiSettings)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    ribbons: RibbonSettings = field(default_factory=RibbonSettings)
    show_objects: bool = True        # the dialog's "show distinct objects" switch, kept for round trips
    warnings: list[str] = field(default_factory=list, compare=False, repr=False)

    # ---- presets ------------------------------------------------------------
    @classmethod
    def default(cls) -> "DetectionConfig":
        return cls()

    @classmethod
    def nanoribbons(cls) -> "DetectionConfig":
        """Preset for graphene nanoribbons: line ROIs along each ribbon and branch lengths."""
        return cls(pipeline=PipelineConfig.from_tuples(NANORIBBON_PIPELINE),
                   roi=RoiSettings(roi_type="LineSegmentROI"),
                   ribbons=RibbonSettings(enabled=True, min_length_px=30.0))

    # ---- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any], registry: StepRegistry = ADAPT_STEPS,
                  strict: bool = False) -> "DetectionConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("a detection config must be a JSON object")
        fmt = data.get("format", PARAM_FILE_FORMAT)
        if fmt != PARAM_FILE_FORMAT:
            raise ConfigError(f"not an image adaptation pipeline file (format {fmt!r})")
        version = data.get("version", PARAM_FILE_VERSION)
        if not isinstance(version, int) or version > PARAM_FILE_VERSION:
            raise ConfigError(f"unsupported file version {version!r} (this build reads up to {PARAM_FILE_VERSION})")
        warnings: list[str] = []
        pipeline = PipelineConfig.from_list(data.get("steps"), registry, warnings)

        roi_raw = dict(data.get("roi") or {})
        show = bool(roi_raw.pop("enabled", True))
        roi_params = {f.name for f in fields(RoiSettings)}
        for k in sorted(set(roi_raw) - roi_params):
            warnings.append(f"roi.{k}: unknown setting, ignored")
        roi = RoiSettings.from_dict(roi_raw).validate()

        labeling = LabelingConfig.from_dict(data.get("labeling"))
        ribbons = RibbonSettings.from_dict(data.get("ribbons"))
        if strict and warnings:
            raise ConfigError("invalid config:\n  " + "\n  ".join(warnings))
        return cls(pipeline, roi, labeling, ribbons, show, warnings)

    def to_dict(self) -> dict[str, Any]:
        return {"format": PARAM_FILE_FORMAT, "version": PARAM_FILE_VERSION,
                "steps": self.pipeline.to_list(),
                "roi": {"enabled": self.show_objects, **self.roi.to_dict()},
                "labeling": self.labeling.to_dict(),
                "ribbons": self.ribbons.to_dict()}

    @classmethod
    def load(cls, path: PathLike, registry: StepRegistry = ADAPT_STEPS, strict: bool = False) -> "DetectionConfig":
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{os.fspath(path)}: not valid JSON ({exc})") from exc
        try:
            return cls.from_dict(data, registry, strict)
        except ConfigError as exc:
            raise type(exc)(f"{os.fspath(path)}: {exc}") from exc

    def save(self, path: PathLike) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
