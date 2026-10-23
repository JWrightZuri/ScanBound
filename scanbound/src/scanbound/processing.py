"""
Reproducible processing chains on calibrated images.

A :class:`ProcessingChain` is a saved list of steps (registry, key, enabled, parameters).
Applying it to an :class:`~scanbound.model.Image` returns a new image whose history lists every
enabled step with its resolved parameters, so the result can be explained and regenerated::

    chain = ProcessingChain.from_dict({"steps": [{"key": "plane"}, {"key": "align_rows",
                                                  "params": {"method": "Median"}}]})
    levelled = chain.apply(raw)                 # Image, values in metres, history extended
    chain.regenerate(levelled)                  # reload the raw file + re-apply -> same pixels

SPM steps keep physical units (``scanbound.steps.spm``).  Samples without data (NaN, e.g. an
incomplete scan) are filled from their nearest valid neighbour for the duration of the chain, so
filters do not smear NaNs, and are NaN again in the result; the history says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .errors import ConfigError
from .model import Image, ProcessingRecord
from .pipeline import Pipeline, PipelineCache
from .steps.registry import StepRegistry
from .steps.spm import SPM_STEPS

CHAIN_FORMAT = "scanbound_processing_chain"
REGISTRIES: dict[str, StepRegistry] = {"spm": SPM_STEPS}


def register_registry(registry: StepRegistry) -> None:
    REGISTRIES[registry.name] = registry


@dataclass
class ChainStep:
    key: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "enabled": self.enabled, "params": dict(self.params)}


@dataclass
class ProcessingChain:
    steps: list[ChainStep] = field(default_factory=list)
    registry: str = "spm"
    nan_policy: str = "nearest"           # how invalid samples are filled while processing
    warnings: list[str] = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.registry not in REGISTRIES:
            raise ConfigError(f"unknown step registry {self.registry!r}")
        reg = REGISTRIES[self.registry]
        for i, s in enumerate(self.steps):
            spec = reg[s.key]                                     # raises UnknownStepError
            s.params = spec.resolve(s.params, self.warnings, where=f"steps[{i}] ({s.key}).")

    # ---- construction / serialisation ------------------------------------------------
    @classmethod
    def from_tuples(cls, steps: list[tuple[str, bool, dict]], registry: str = "spm") -> "ProcessingChain":
        return cls([ChainStep(k, e, dict(p)) for k, e, p in steps], registry)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "ProcessingChain":
        d = d or {}
        fmt = d.get("format", CHAIN_FORMAT)
        if fmt not in (CHAIN_FORMAT, "scanbound_spm_pipeline"):
            raise ConfigError(f"not a processing chain (format {fmt!r})")
        raw = d.get("steps") or []
        if not isinstance(raw, list):
            raise ConfigError("'steps' must be a list")
        steps = []
        for i, e in enumerate(raw):
            if not isinstance(e, Mapping):
                raise ConfigError(f"steps[{i}] must be an object")
            key = e.get("key", e.get("type"))
            steps.append(ChainStep(str(key), bool(e.get("enabled", True)), dict(e.get("params") or {})))
        return cls(steps, str(d.get("registry", "spm")), str(d.get("nan_policy", "nearest")))

    def to_dict(self) -> dict[str, Any]:
        return {"format": CHAIN_FORMAT, "version": 1, "registry": self.registry, "nan_policy": self.nan_policy,
                "steps": [s.to_dict() for s in self.steps]}

    def copy(self) -> "ProcessingChain":
        return ProcessingChain.from_dict(self.to_dict())

    @property
    def active(self) -> list[ChainStep]:
        return [s for s in self.steps if s.enabled]

    def __bool__(self) -> bool:
        return bool(self.active)

    def summary(self) -> str:
        reg = REGISTRIES[self.registry]
        return ", ".join(reg[s.key].title for s in self.active) or "raw data"

    def pipeline(self) -> Pipeline:
        return Pipeline([(s.key, s.enabled, s.params) for s in self.steps], REGISTRIES[self.registry])

    # ---- running -----------------------------------------------------------------------------
    def apply(self, image: Image, cache: Optional[PipelineCache] = None, strict: bool = True) -> Image:
        """The processed image.  ``strict`` (default) raises on a failing step; otherwise the step is
        skipped and recorded with its error."""
        if not self.active:
            return image
        bad = ~image.valid
        src = image.filled(self.nan_policy) if bad.any() else np.asarray(image.data, dtype=np.float64)
        if cache is not None:
            # cache entries are bound to the identity of the source array: keep one per image
            key = getattr(cache, "_sb_source", None)
            if key is None or key[0] != image.digest:
                cache._sb_source = (image.digest, src)   # type: ignore[attr-defined]
            src = cache._sb_source[1]                    # type: ignore[attr-defined]
        res = self.pipeline().run(src, strict=strict, cache=cache)
        out = np.array(res.image, dtype=np.float64)
        if bad.any():
            out[bad] = np.nan
        failed = {e.index: e.message for e in res.errors}
        records = []
        reg = REGISTRIES[self.registry]
        for i, s in enumerate(self.steps):
            if not s.enabled:
                continue
            note = f"FAILED, skipped: {failed[i]}" if i in failed else ""
            records.append(ProcessingRecord(f"{self.registry}.{s.key}", s.params, "values",
                                            note or reg[s.key].title))
        if bad.any():
            records.append(ProcessingRecord("nan_policy", {"fill": self.nan_policy}, "none",
                                            f"{int(bad.sum())} invalid samples filled while processing, NaN again "
                                            "in the result"))
        return image.derive(out, records[0]).appended(*records[1:])

    @staticmethod
    def from_history(image: Image) -> "ProcessingChain":
        """The chain that produced ``image`` (its processing records), for regeneration."""
        steps = []
        registry = "spm"
        for r in image.history:
            reg, _, key = r.op.partition(".")
            if reg in REGISTRIES and key and r.effect == "values" and not r.note.startswith("FAILED"):
                registry = reg
                steps.append(ChainStep(key, True, dict(r.params)))
        return ProcessingChain(steps, registry)

    @staticmethod
    def regenerate(image: Image) -> Image:
        """Reload ``image``'s raw source and re-apply its recorded processing (and calibration)."""
        from .io import open_data
        if not image.source.path:
            raise ConfigError("the image has no source file to regenerate from")
        raw = open_data(image.source.path).image(image.source.channel or None)
        for r in image.history:
            if r.op == "calibration.set":
                p = r.params
                from .model import Calibration
                raw = raw.with_calibration(p.get("dx"), p.get("dy"), unit=p.get("unit", "m"),
                                           calibration=Calibration(p.get("source", "manual"), p.get("detail", ""),
                                                                   float(p.get("rel_uncertainty", 0.0))))
        return ProcessingChain.from_history(image).apply(raw)
