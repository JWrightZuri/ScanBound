"""
Step and parameter descriptions, and the registry that pipelines look steps up in.

A step is a plain function ``func(image, **params) -> image`` plus a :class:`StepSpec` that
describes its parameters.  The descriptions carry everything a UI needs to build editors
(label, range, step, decimals, choices, visibility rule, tooltip), but nothing here imports Qt:
the PyQt dialogs read these objects, the headless detector only uses :meth:`StepSpec.resolve`.

Adding a step::

    from scanbound.steps import ADAPT_STEPS, Param, StepSpec

    def f_my_filter(img, radius):
        ...

    ADAPT_STEPS.register(StepSpec("my_filter", "My filter", "gray", f_my_filter, (
        Param("radius", "Radius (px)", "float", 2.0, 0.5, 20, 0.5, 1),
    ), "What it does, shown as the card's tooltip."))
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ..errors import ParamError, UnknownStepError

ParamKind = Literal["float", "int", "odd", "choice", "bool"]
ImageKind = Literal["gray", "binary"]
StepFunc = Callable[..., Any]


@dataclass(frozen=True)
class Param:
    """One parameter of a step.  Field order matches the legacy ``core`` class."""

    key: str
    label: str
    kind: ParamKind
    default: Any
    minimum: float = 0
    maximum: float = 1
    step: float = 1
    decimals: int = 2
    choices: tuple[str, ...] = ()
    visible_if: Optional[Callable[[dict], bool]] = None
    tooltip: str = ""

    def is_visible(self, params: Mapping[str, Any]) -> bool:
        return self.visible_if is None or bool(self.visible_if(dict(params)))

    def coerce(self, value: Any, warnings: Optional[list[str]] = None, where: str = "") -> Any:
        """
        ``value`` converted to this parameter's type and range.  Out-of-range numbers are clamped
        and an unknown choice falls back to the default; both add a message to ``warnings``.
        Values of the wrong type raise :class:`ParamError`.
        """
        name = f"{where}{self.key}"

        def warn(msg: str) -> None:
            if warnings is not None:
                warnings.append(f"{name}: {msg}")

        if self.kind == "bool":
            if isinstance(value, str):
                if value.strip().lower() in ("true", "1", "yes", "on"):
                    return True
                if value.strip().lower() in ("false", "0", "no", "off", ""):
                    return False
                raise ParamError(f"{name}: expected a boolean, got {value!r}")
            return bool(value)
        if self.kind == "choice":
            if str(value) in self.choices:
                return str(value)
            warn(f"unknown choice {value!r}, using {self.default!r}")
            return self.default
        if isinstance(value, bool) or value is None:
            raise ParamError(f"{name}: expected a number, got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ParamError(f"{name}: expected a number, got {value!r}") from None
        if not math.isfinite(number):
            raise ParamError(f"{name}: expected a finite number, got {value!r}")
        clamped = min(max(number, float(self.minimum)), float(self.maximum))
        if clamped != number:
            warn(f"{value!r} is outside [{self.minimum}, {self.maximum}], clamped to {clamped:g}")
        if self.kind == "float":
            return clamped
        whole = int(round(clamped))
        if self.kind == "odd" and whole % 2 == 0:
            whole = whole + 1 if whole + 1 <= self.maximum else whole - 1
            warn(f"must be odd, using {whole}")
        return whole


@dataclass(frozen=True)
class StepSpec:
    """
    A processing step.

    ``needs`` is the kind of image the step takes: ``"gray"`` steps get a float image (in [0, 1]
    for the adaptation steps, in physical units for the SPM steps), ``"binary"`` steps get a bool
    mask.  A gray step that returns a bool array (the threshold) turns the pipeline binary.
    ``extra_keys`` names keyword arguments the function takes that have no editor (for example
    the shapes of the Fourier filter, drawn in their own dialog).
    """

    key: str
    title: str
    needs: ImageKind
    func: StepFunc
    params: Sequence[Param] = ()
    description: str = ""
    extra_keys: Sequence[str] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "extra_keys", tuple(self.extra_keys))

    def defaults(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params}

    def param(self, key: str) -> Param:
        for p in self.params:
            if p.key == key:
                return p
        raise KeyError(key)

    def resolve(self, overrides: Optional[Mapping[str, Any]] = None,
                warnings: Optional[list[str]] = None, where: str = "") -> dict[str, Any]:
        """Defaults updated with ``overrides``, each value validated by its :class:`Param`.
        Unknown keys are dropped with a warning."""
        values = self.defaults()
        known = {p.key: p for p in self.params}
        for k, v in (overrides or {}).items():
            if k in known:
                values[k] = known[k].coerce(v, warnings, where)
            elif k in self.extra_keys:
                values[k] = v
            elif warnings is not None:
                warnings.append(f"{where}{k}: not a parameter of {self.key!r}, ignored")
        return values

    def __call__(self, image: Any, **params: Any) -> Any:
        return self.func(image, **params)


class StepRegistry(Mapping[str, StepSpec]):
    """An ordered, name-addressed collection of steps (behaves like a read-only dict)."""

    def __init__(self, name: str, steps: tuple[StepSpec, ...] | list[StepSpec] = ()):
        self.name = name
        self._steps: dict[str, StepSpec] = {}
        for s in steps:
            self.register(s)

    def register(self, spec: StepSpec, replace: bool = False) -> StepSpec:
        if spec.key in self._steps and not replace:
            raise ValueError(f"step {spec.key!r} is already registered in {self.name!r}")
        self._steps[spec.key] = spec
        return spec

    def __getitem__(self, key: str) -> StepSpec:
        try:
            return self._steps[key]
        except KeyError:
            raise UnknownStepError(
                f"unknown step {key!r} for the {self.name} pipeline; known steps: {', '.join(self._steps)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"StepRegistry({self.name!r}, {list(self._steps)})"

    def extended(self, name: str, *steps: StepSpec) -> "StepRegistry":
        """A copy with more steps (for building one registry on top of another)."""
        return StepRegistry(name, [*self._steps.values(), *steps])
