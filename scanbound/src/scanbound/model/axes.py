"""
Axes and calibration.

An :class:`Axis` says what one array dimension means: its name, its unit, how many samples it
has and where each sample sits in physical coordinates.  Its :class:`Calibration` says where that
scale came from and how far it can be trusted, so every number derived from it can explain itself.

Coordinate convention for image axes (shared with the annotations): a continuous pixel
coordinate ``u`` has 0 at the *outer edge* of the first pixel, so pixel ``i`` covers
``[i, i + 1]`` and its centre is ``i + 0.5``.  The physical coordinate is ``offset + u * step``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np

from .units import format_value

# where a scale came from, from most to least trustworthy
CALIBRATION_SOURCES = ("file", "manual", "sidecar", "assumed", "none")


@dataclass(frozen=True)
class Calibration:
    """
    Origin and trust of an axis scale.

    ``source``: ``"file"`` (read from the instrument's header), ``"manual"`` (typed in by a
    person), ``"sidecar"`` (a separate file such as ScanBound's old ``metadata.csv``),
    ``"assumed"`` (a default the software chose) or ``"none"`` (pixels only).
    ``rel_uncertainty`` is the relative standard uncertainty of the step (0.02 = 2 %); it is an
    *assumption* unless measured against a reference, and ``detail`` should say which.
    """

    source: str = "none"
    detail: str = ""
    rel_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in CALIBRATION_SOURCES:
            raise ValueError(f"unknown calibration source {self.source!r}; expected one of {CALIBRATION_SOURCES}")
        if not (self.rel_uncertainty >= 0):
            raise ValueError("rel_uncertainty must be >= 0")

    @property
    def calibrated(self) -> bool:
        return self.source != "none"

    def describe(self) -> str:
        if not self.calibrated:
            return "not calibrated (pixel units)"
        what = {"file": "from the file header", "manual": "entered manually", "sidecar": "from a sidecar file",
                "assumed": "assumed"}[self.source]
        unc = f", ±{self.rel_uncertainty * 100:.3g} % (1σ)" if self.rel_uncertainty else ", uncertainty not specified"
        return what + (f" ({self.detail})" if self.detail else "") + unc

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "detail": self.detail, "rel_uncertainty": self.rel_uncertainty}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Calibration":
        d = d or {}
        return cls(str(d.get("source", "none")), str(d.get("detail", "")), float(d.get("rel_uncertainty", 0.0)))


UNCALIBRATED = Calibration()

# Default relative uncertainty of a piezo scanner calibration when nothing better is known.  It is
# an assumption and is always reported as one: typical STM/AFM lateral calibrations drift by a
# few per cent between calibrations.
DEFAULT_SCANNER_REL_UNCERTAINTY = 0.02


@dataclass(frozen=True)
class Axis:
    """
    One dimension of a data array.

    Regular axes have ``step`` (physical size of one sample) and ``offset`` (coordinate of the
    edge of sample 0).  Irregular axes, such as a non-linear bias sweep, list every sample's
    coordinate in ``values`` instead.  An uncalibrated spatial axis has ``step=None`` and works in
    pixels.
    """

    name: str
    size: int
    unit: str = ""
    step: Optional[float] = None
    offset: float = 0.0
    values: Optional[tuple[float, ...]] = None
    calibration: Calibration = field(default=UNCALIBRATED)
    label: str = ""                      # human name, e.g. "Bias"

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("axis size must be >= 0")
        if self.values is not None:
            vals = tuple(float(v) for v in self.values)
            if len(vals) != self.size:
                raise ValueError(f"axis {self.name!r}: {len(vals)} values for size {self.size}")
            object.__setattr__(self, "values", vals)
        if self.step is not None:
            step = float(self.step)
            if not np.isfinite(step) or step == 0:
                raise ValueError(f"axis {self.name!r}: step must be finite and non-zero, got {self.step!r}")
            object.__setattr__(self, "step", step)

    # ---- constructors ---------------------------------------------------------
    @classmethod
    def pixels(cls, name: str, size: int) -> "Axis":
        return cls(name, size, "px")

    @classmethod
    def spatial(cls, name: str, size: int, step_m: Optional[float], offset_m: float = 0.0,
                calibration: Optional[Calibration] = None) -> "Axis":
        """A length axis in metres (``step_m=None`` -> uncalibrated pixel axis)."""
        if step_m is None or not np.isfinite(step_m) or step_m <= 0:
            return cls.pixels(name, size)
        return cls(name, size, "m", float(step_m), float(offset_m), calibration=calibration or Calibration("assumed"))

    @classmethod
    def from_values(cls, name: str, values: Sequence[float], unit: str = "", label: str = "",
                    calibration: Optional[Calibration] = None) -> "Axis":
        vals = np.asarray(values, float).ravel()
        return cls(name, int(vals.size), unit, values=tuple(vals.tolist()), label=label,
                   calibration=calibration or Calibration("file"))

    # ---- properties -------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return (self.step is not None or self.values is not None) and self.unit not in ("", "px")

    @property
    def regular(self) -> bool:
        return self.values is None

    @property
    def scale(self) -> float:
        """Physical size per sample, 1.0 for pixel axes."""
        if self.step is not None:
            return self.step
        if self.values is not None and self.size >= 2:
            return (self.values[-1] - self.values[0]) / (self.size - 1)
        return 1.0

    @property
    def length(self) -> float:
        """Physical extent (size * step), or size in pixels."""
        return self.size * abs(self.scale)

    def coords(self, centres: bool = True) -> np.ndarray:
        """Coordinate of every sample (pixel centres by default)."""
        if self.values is not None:
            return np.asarray(self.values, float)
        u = np.arange(self.size, dtype=float) + (0.5 if centres else 0.0)
        return self.offset + u * self.scale if self.step is not None else u

    def to_physical(self, u: Any) -> Any:
        """Continuous pixel coordinate(s) -> physical coordinate(s)."""
        u = np.asarray(u, float)
        if self.values is not None:
            idx = np.arange(self.size, dtype=float) + 0.5
            return np.interp(u, idx, np.asarray(self.values, float))
        if self.step is None:
            return u
        return self.offset + u * self.step

    def to_pixel(self, x: Any) -> Any:
        x = np.asarray(x, float)
        if self.values is not None:
            vals = np.asarray(self.values, float)
            order = np.argsort(vals)
            return np.interp(x, vals[order], (np.arange(self.size, dtype=float) + 0.5)[order])
        if self.step is None:
            return x
        return (x - self.offset) / self.step

    def with_calibration(self, step: Optional[float], calibration: Calibration, unit: str = "m",
                         offset: Optional[float] = None) -> "Axis":
        if step is None:
            return replace(self, unit="px", step=None, offset=0.0, calibration=UNCALIBRATED)
        return replace(self, unit=unit, step=float(step), offset=self.offset if offset is None else offset,
                       calibration=calibration)

    def resized(self, size: int, start: float = 0.0, factor: float = 1.0) -> "Axis":
        """Axis of a cropped (``start``, in pixels) or resampled (``factor`` new samples per old) array."""
        if self.values is not None:
            idx = start + (np.arange(size) + 0.5) / factor - 0.5
            return replace(self, size=size, values=tuple(np.interp(idx, np.arange(self.size), self.values).tolist()))
        if self.step is None:
            return replace(self, size=size)
        return replace(self, size=size, step=self.step / factor, offset=self.offset + start * self.step)

    def describe(self) -> str:
        name = self.label or self.name
        if not self.calibrated:
            return f"{name}: {self.size} px, {self.calibration.describe()}"
        if self.values is not None:
            return (f"{name}: {self.size} samples from {format_value(self.values[0], self.unit)} to "
                    f"{format_value(self.values[-1], self.unit)}")
        return (f"{name}: {self.size} × {format_value(self.step, self.unit)} = "
                f"{format_value(self.length, self.unit)}, {self.calibration.describe()}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "size": self.size, "unit": self.unit}
        if self.step is not None:
            d.update(step=self.step, offset=self.offset)
        if self.values is not None:
            d["values"] = list(self.values)
        if self.label:
            d["label"] = self.label
        d["calibration"] = self.calibration.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Axis":
        return cls(str(d["name"]), int(d["size"]), str(d.get("unit", "")), d.get("step"), float(d.get("offset", 0.0)),
                   tuple(d["values"]) if d.get("values") is not None else None,
                   Calibration.from_dict(d.get("calibration")), str(d.get("label", "")))
