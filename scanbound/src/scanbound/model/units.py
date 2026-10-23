"""
Units, SI prefixes and quantities with uncertainty.

Data are stored in base SI units (m, V, A, s, ...) exactly as the instrument wrote them; prefixes
only appear when values are shown to people.  The few helpers here are deliberately small: they
know SI prefixes and a handful of aliases, which is all the supported file formats need.

    >>> to_base(12.5, "nm")
    (1.25e-08, 'm')
    >>> format_value(1.25e-08, "m")
    '12.5 nm'
    >>> Quantity(1.25e-8, "m", 4e-10).format()
    '12.50 ± 0.40 nm'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

PREFIXES = {"y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
            "μ": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_DISPLAY_PREFIXES = [(1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, ""),
                     (1e3, "k"), (1e6, "M"), (1e9, "G")]
BASE_UNITS = ("m", "V", "A", "s", "Hz", "N", "K", "T", "W", "Pa", "S", "F", "eV", "g")
_ALIASES = {"Å": (1e-10, "m"), "Ang": (1e-10, "m"), "angstrom": (1e-10, "m"), "micron": (1e-6, "m"),
            "deg": (1.0, "°"), "degree": (1.0, "°"), "degrees": (1.0, "°"), "°": (1.0, "°"),
            "rad": (180.0 / math.pi, "°"), "%": (1.0, "%"), "px": (1.0, "px")}
# units that never take an SI prefix when displayed
_NO_PREFIX = {"", "°", "%", "px", "K", "a.u.", "arb", "counts", "count"}


def split_unit(unit: str) -> tuple[float, str]:
    """``"nm" -> (1e-9, "m")``, ``"pA" -> (1e-12, "A")``; unknown units come back unchanged with factor 1."""
    u = (unit or "").strip()
    if u in _ALIASES:
        return _ALIASES[u]
    if u in BASE_UNITS or u in _NO_PREFIX:
        return 1.0, u
    for base in sorted(BASE_UNITS, key=len, reverse=True):
        if u.endswith(base) and u[: -len(base)] in PREFIXES:
            return PREFIXES[u[: -len(base)]], base
    return 1.0, u


def to_base(value: float, unit: str) -> tuple[float, str]:
    """``value`` in ``unit`` converted to the base unit."""
    factor, base = split_unit(unit)
    return value * factor, base


def convert(value: float, unit: str, target: str) -> float:
    """Convert between two units of the same dimension (``convert(1, "nm", "Å") == 10``)."""
    f1, b1 = split_unit(unit)
    f2, b2 = split_unit(target)
    if b1 != b2:
        raise ValueError(f"cannot convert {unit!r} to {target!r}")
    return value * f1 / f2


def is_length(unit: str) -> bool:
    return split_unit(unit)[1] == "m"


def best_prefix(value: float, unit: str) -> tuple[float, str]:
    """(divisor, prefixed unit) that shows ``value`` (in base ``unit``) with 1-3 integer digits."""
    if unit in _NO_PREFIX or not math.isfinite(value) or value == 0:
        return 1.0, unit
    scale, prefix = _DISPLAY_PREFIXES[0]
    for s, p in _DISPLAY_PREFIXES:
        if abs(value) >= s * 0.9999:
            scale, prefix = s, p
    return scale, prefix + unit


def format_value(value: Optional[float], unit: str = "", digits: int = 4) -> str:
    """``format_value(1e-10, "A") -> "100 pA"``.  ``None`` / NaN give ``"n/a"``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    scale, u = best_prefix(float(value), unit)
    return f"{float(value) / scale:.{digits}g} {u}".strip()


@dataclass(frozen=True)
class Quantity:
    """A value in a base unit with an optional standard uncertainty (same unit)."""

    value: float
    unit: str = ""
    uncertainty: Optional[float] = None

    @property
    def relative_uncertainty(self) -> Optional[float]:
        if self.uncertainty is None or self.value == 0:
            return None
        return abs(self.uncertainty / self.value)

    def to(self, unit: str) -> float:
        return convert(self.value, self.unit, unit)

    def format(self, digits: int = 2) -> str:
        """Value with the uncertainty rounded to ``digits`` significant digits: ``12.5 ± 0.4 nm``."""
        if self.uncertainty is None or not math.isfinite(self.uncertainty) or self.uncertainty <= 0:
            return format_value(self.value, self.unit)
        scale, u = best_prefix(self.value if self.value else self.uncertainty, self.unit)
        v, e = self.value / scale, self.uncertainty / scale
        decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(e)))))
        return f"{v:.{decimals}f} ± {e:.{decimals}f} {u}".strip()

    def __str__(self) -> str:
        return self.format()

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "uncertainty": self.uncertainty}
