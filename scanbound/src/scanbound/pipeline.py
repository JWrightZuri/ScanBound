"""
Running an ordered list of steps on an image.

    from scanbound import Pipeline
    pipe = Pipeline.from_config(config.pipeline)            # or Pipeline([("gaussian", True, {...}), ...])
    result = pipe.run(image)                                 # PipelineResult
    result.image, result.errors, result.is_mask

A failing step is skipped and reported in ``result.errors`` (so an interactive editor stays
usable); ``strict=True`` raises :class:`PipelineError` instead, which is what batch runs want.

:class:`PipelineCache` keeps every step's output; when only a later step changes, the steps
before it are not recomputed.  That is what makes slider dragging in the dialogs fast.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .errors import PipelineError
from .steps.adapt import ADAPT_STEPS
from .steps.registry import StepRegistry, StepSpec

if TYPE_CHECKING:
    from .config import PipelineConfig

StepLike = Union[tuple[str, bool, Mapping[str, Any]], tuple[StepSpec, Mapping[str, Any], bool]]


@dataclass(frozen=True)
class BoundStep:
    """A step with resolved parameters, ready to run."""

    spec: StepSpec
    params: Mapping[str, Any]
    enabled: bool = True

    @property
    def signature(self) -> tuple:
        return (self.spec.key, _freeze(self.params), bool(self.enabled))


@dataclass(frozen=True)
class StepFailure:
    index: int
    step: str
    message: str


@dataclass
class PipelineResult:
    image: np.ndarray
    errors: list[StepFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def is_mask(self) -> bool:
        return self.image.dtype == np.bool_

    def errors_by_index(self) -> dict[int, str]:
        """``{step index: message}``, the format of the legacy ``run_pipeline``."""
        return {e.index: e.message for e in self.errors}


class PipelineCache:
    """Per-step outputs of the last run, valid for one source image."""

    def __init__(self) -> None:
        self._source: Optional[tuple] = None
        self._entries: list[tuple[tuple, np.ndarray, Optional[str]]] = []

    def clear(self) -> None:
        self._source = None
        self._entries.clear()

    def _bind(self, source: np.ndarray) -> list[tuple[tuple, np.ndarray, Optional[str]]]:
        key = (id(source), source.shape, source.dtype.str)
        if key != self._source:
            self._source = key
            self._entries.clear()
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)


def _freeze(value: Any) -> Any:
    """Hashable, comparable version of nested params (lists from JSON -> tuples)."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return (value.shape, value.tobytes())
    return value


class Pipeline:
    """An ordered list of :class:`BoundStep`."""

    def __init__(self, steps: Iterable[StepLike | BoundStep] = (), registry: StepRegistry = ADAPT_STEPS,
                 warnings: Optional[list[str]] = None):
        self.registry = registry
        self.steps: list[BoundStep] = [self._bind(s, i, warnings) for i, s in enumerate(steps)]

    def _bind(self, step: StepLike | BoundStep, index: int, warnings: Optional[list[str]]) -> BoundStep:
        if isinstance(step, BoundStep):
            return step
        first, second, third = step
        if isinstance(first, StepSpec):                      # legacy (spec, params, enabled)
            assert isinstance(second, Mapping)
            return BoundStep(first, dict(second), bool(third))
        spec = self.registry[str(first)]
        assert isinstance(third, Mapping) or third is None
        params = spec.resolve(third, warnings, where=f"step {index} ({spec.key}).")
        return BoundStep(spec, params, bool(second))

    @classmethod
    def from_config(cls, config: "PipelineConfig", registry: StepRegistry = ADAPT_STEPS,
                    warnings: Optional[list[str]] = None) -> "Pipeline":
        return cls([(s.type, s.enabled, s.params) for s in config.steps], registry, warnings)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self) -> str:
        names = [s.spec.key + ("" if s.enabled else " (off)") for s in self.steps]
        return f"Pipeline({self.registry.name}: {' -> '.join(names) or 'empty'})"

    @property
    def produces_mask(self) -> bool:
        """True if an enabled step converts gray -> binary (a threshold) or works on binary."""
        return any(s.enabled and (s.spec.key == "threshold" or s.spec.needs == "binary") for s in self.steps)

    def run(self, source: np.ndarray, *, strict: bool = False,
            cache: Optional[PipelineCache] = None) -> PipelineResult:
        cur = np.asarray(source)
        errors: list[StepFailure] = []
        entries = cache._bind(source) if cache is not None else None
        reuse = entries is not None
        for i, step in enumerate(self.steps):
            sig = step.signature
            if entries is not None and reuse and i < len(entries) and entries[i][0] == sig:
                cur, err = entries[i][1], entries[i][2]
                if err:
                    errors.append(StepFailure(i, step.spec.key, err))
                continue
            reuse = False
            err = None
            if step.enabled:
                is_bin = cur.dtype == np.bool_
                if step.spec.needs == "binary" and not is_bin:
                    err = "Needs a binary image: put a Threshold step above this one."
                else:
                    inp = cur.astype(np.float32) if step.spec.needs == "gray" and is_bin else cur
                    try:
                        cur = np.asarray(step.spec.func(inp, **step.params))
                    except Exception as exc:  # noqa: BLE001 - reported per step
                        err = f"{type(exc).__name__}: {exc}"
                        if strict:
                            raise PipelineError(i, step.spec.key, err) from exc
                        cur = inp
                if err and strict:
                    raise PipelineError(i, step.spec.key, err)
            if err:
                errors.append(StepFailure(i, step.spec.key, err))
            if entries is not None:
                del entries[i:]
                entries.append((sig, cur, err))
        if entries is not None:
            del entries[len(self.steps):]
        return PipelineResult(cur, errors)


def run_pipeline(source: np.ndarray, steps: Sequence[tuple[StepSpec, Mapping[str, Any], bool]],
                 cache: Optional[list] = None) -> tuple[np.ndarray, dict[int, str]]:
    """
    Drop-in replacement for ``core.image_adapt_dialog.run_pipeline``.

    ``steps`` is a list of ``(StepSpec, params, enabled)``; returns ``(image, {index: error})``.
    ``cache`` is a list kept by the caller, as before.
    """
    pipe = Pipeline([BoundStep(spec, dict(params), bool(enabled)) for spec, params, enabled in steps])
    if cache is None:
        res = pipe.run(source)
        return res.image, res.errors_by_index()
    # adapt the legacy list cache: entries are (signature, image, error)
    shim = PipelineCache()
    shim._source = (id(source), np.shape(source), np.asarray(source).dtype.str)
    shim._entries = cache
    res = pipe.run(source, cache=shim)
    return res.image, res.errors_by_index()
