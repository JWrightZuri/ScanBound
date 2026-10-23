"""
Point spectroscopy (STS, force-distance, Z spectroscopy ...).

A :class:`Spectrum` is a set of signals sampled on one sweep axis (bias, Z, time), measured at
one tip position.  Signals keep their own units.  Repeated sweeps and backward sweeps are kept as
separate signals named like the instrument names them (``"Current (A) [bwd]"``,
``"Current [00002]"``), with :meth:`Spectrum.signal` looking them up by base name.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .axes import Axis
from .provenance import History, ProcessingRecord, SourceRef

_UNIT_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<unit>[^()]*)\)\s*(?P<tag>(\[[^\]]*\]\s*)*)$")


def split_column(title: str) -> tuple[str, str, str]:
    """``"Current (A) [bwd]" -> ("Current", "A", "[bwd]")``; titles without a unit give ``""``."""
    m = _UNIT_RE.match(title.strip())
    if not m:
        tags = " ".join(re.findall(r"\[[^\]]*\]", title))
        return re.sub(r"\s*\[[^\]]*\]", "", title).strip(), "", tags
    return m.group("name").strip(), m.group("unit").strip(), m.group("tag").strip()


@dataclass(frozen=True)
class Signal:
    name: str                 # base name, e.g. "Current"
    unit: str
    data: np.ndarray
    tag: str = ""             # "[bwd]", "[00002]", "[filt]" ...

    @property
    def key(self) -> str:
        return f"{self.name} ({self.unit})" + (f" {self.tag}" if self.tag else "")

    @property
    def backward(self) -> bool:
        return "bwd" in self.tag.lower()


@dataclass
class Spectrum:
    axis: Axis                                 # sweep axis with explicit values
    signals: list[Signal]
    position: Optional[tuple[float, float, float]] = None   # tip x, y, z in m (sample frame)
    kind: str = ""                             # "bias spectroscopy", "Z spectroscopy" ...
    source: SourceRef = field(default_factory=SourceRef)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: History = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        for s in self.signals:
            if len(s.data) != self.axis.size:
                raise ValueError(f"signal {s.key!r} has {len(s.data)} samples, axis has {self.axis.size}")

    @property
    def x(self) -> np.ndarray:
        return self.axis.coords()

    def __iter__(self) -> Iterator[Signal]:
        return iter(self.signals)

    def names(self) -> list[str]:
        return [s.key for s in self.signals]

    def signal(self, name: str, tag: Optional[str] = "") -> Optional[Signal]:
        """By full key (``"Current (A)"``) or base name; ``tag=None`` accepts any tag."""
        for s in self.signals:
            if s.key == name:
                return s
        for s in self.signals:
            if s.name == name and (tag is None or s.tag == tag):
                return s
        return None

    def with_signal(self, signal: Signal, record: ProcessingRecord) -> "Spectrum":
        return Spectrum(self.axis, [*self.signals, signal], self.position, self.kind, self.source, self.metadata,
                        self.history + (record,), self.id)

    def describe(self) -> str:
        from .units import format_value
        lo, hi = self.axis.values[0], self.axis.values[-1]  # type: ignore[index]
        where = ""
        if self.position:
            where = f" at ({format_value(self.position[0], 'm')}, {format_value(self.position[1], 'm')})"
        return (f"{self.kind or 'spectrum'}: {self.axis.size} points, {self.axis.label or self.axis.name} "
                f"{format_value(lo, self.axis.unit)} … {format_value(hi, self.axis.unit)}{where}")


def average(spectra: Sequence[Spectrum], signal: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error of one signal over spectra on the same axis: (x, mean, sem)."""
    if not spectra:
        raise ValueError("no spectra")
    x = spectra[0].x
    rows = []
    for sp in spectra:
        s = sp.signal(signal)
        if s is None:
            raise KeyError(f"{sp.source.describe()}: no signal {signal!r}")
        if sp.axis.size != len(x) or not np.allclose(sp.x, x, rtol=1e-6, atol=1e-12):
            rows.append(np.interp(x, sp.x, s.data) if np.all(np.diff(sp.x) > 0) else np.interp(x, sp.x[::-1], s.data[::-1]))
        else:
            rows.append(s.data)
    arr = np.vstack(rows)
    n = np.sum(np.isfinite(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1 if len(rows) > 1 else 0) / np.sqrt(np.maximum(n, 1))
    return x, mean, sem
