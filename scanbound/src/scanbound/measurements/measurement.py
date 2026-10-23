"""
A measurement is a number that can explain itself.

:class:`Measurement` holds a value, its unit and standard uncertainty, *how* it was obtained
(``method``), what had to be assumed (``assumptions``) and what it was computed from (``inputs``:
annotation ids, image id, calibration).  Tables of measurements export with all of it, so a
length in a paper can be traced back to the pixels, the calibration and the person or detector
that drew the line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..model.units import Quantity, format_value


def format_amount(value: float, unit: str, digits: int = 4) -> str:
    """A value in a base unit, with a prefix; areas (``m²``) get a squared prefix (``nm²``)."""
    if unit.endswith("²") and len(unit) > 1:
        from ..model.units import best_prefix
        scale, pu = best_prefix(math.sqrt(abs(value)) if value else 1.0, unit[:-1])
        return f"{value / scale ** 2:.{digits}g} {pu}²"
    if unit == "°":
        return f"{value:.{digits}g}°"
    return format_value(value, unit, digits)


@dataclass(frozen=True)
class Measurement:
    name: str                       # "length", "area", "angle", "step_height" ...
    value: float
    unit: str = ""
    uncertainty: Optional[float] = None
    method: str = ""
    assumptions: tuple[str, ...] = ()
    inputs: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)   # uncertainty budget: source -> contribution (same unit)

    @property
    def quantity(self) -> Quantity:
        return Quantity(self.value, self.unit, self.uncertainty)

    @property
    def relative_uncertainty(self) -> Optional[float]:
        return self.quantity.relative_uncertainty

    def format(self, digits: int = 2) -> str:
        if self.unit == "°":
            if self.uncertainty is None:
                return f"{self.value:.4g}°"
            d = max(0, digits - 1 - int(math.floor(math.log10(self.uncertainty)))) if self.uncertainty > 0 else 1
            return f"{self.value:.{d}f} ± {self.uncertainty:.{d}f}°"
        if not self.unit or self.unit in ("px", "px²", "%"):
            if self.uncertainty:
                d = max(0, digits - 1 - int(math.floor(math.log10(self.uncertainty))))
                return f"{self.value:.{d}f} ± {self.uncertainty:.{d}f} {self.unit}".strip()
            return f"{self.value:.4g} {self.unit}".strip()
        if self.unit.endswith("²"):
            base = self.unit[:-1]
            # area: show in (prefix base)², e.g. nm²
            from ..model.units import best_prefix
            scale, pu = best_prefix(math.sqrt(abs(self.value)) if self.value else 1.0, base)
            v = self.value / scale ** 2
            if self.uncertainty:
                e = self.uncertainty / scale ** 2
                d = max(0, digits - 1 - int(math.floor(math.log10(e)))) if e > 0 else 2
                return f"{v:.{d}f} ± {e:.{d}f} {pu}²"
            return f"{v:.4g} {pu}²"
        return self.quantity.format(digits) if self.uncertainty else format_value(self.value, self.unit)

    def explain(self) -> str:
        lines = [f"{self.name} = {self.format()}"]
        if self.method:
            lines.append(f"Method: {self.method}")
        if self.components:
            budget = ", ".join(f"{k} {format_amount(v, self.unit, 3)}" for k, v in self.components.items())
            lines.append(f"Uncertainty (1σ, combined in quadrature): {budget}")
        for a in self.assumptions:
            lines.append(f"Assumes: {a}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "uncertainty": self.uncertainty,
                "method": self.method, "assumptions": list(self.assumptions), "inputs": self.inputs,
                "components": self.components}


def combine(components: dict[str, float]) -> float:
    """Root-sum-square of independent uncertainty contributions."""
    return math.sqrt(sum(v * v for v in components.values() if v is not None and math.isfinite(v)))


def table(measurements: Iterable[Measurement], **extra: Any) -> list[dict[str, Any]]:
    """Flat rows (for pandas / CSV): one per measurement, with ``extra`` columns prepended."""
    rows = []
    for m in measurements:
        rows.append({**extra, "measurement": m.name, "value": m.value, "unit": m.unit,
                     "uncertainty": m.uncertainty, "method": m.method, "assumptions": "; ".join(m.assumptions)})
    return rows
