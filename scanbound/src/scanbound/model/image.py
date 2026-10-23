"""
Images as scientific data.

An :class:`Image` is a 2D array of values *with meaning*: the quantity it measures and its unit
(``Z`` in ``m``, ``Current`` in ``A`` ...), a calibrated :class:`~scanbound.model.axes.Axis` for
rows and columns (pixels may be anisotropic), where it came from and every operation applied
since.  Pixel values are never rescaled to 8 bits or colour mapped here: that is the display's
job.  Invalid samples (incomplete scan lines, dropouts) are NaN.

Images are immutable: an operation returns a new image whose ``history`` has one more record.

    img = Image(z, x=Axis.spatial("x", 512, 0.2e-9), y=Axis.spatial("y", 512, 0.2e-9),
                quantity="Z", unit="m")
    img.pixel_size            # (dx, dy) in m
    img.to_physical(points)   # pixel (x, y) -> metres
    img.quality()             # NaNs, saturation, anisotropy, calibration ... as warnings
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .axes import Axis, Calibration
from .provenance import History, ProcessingRecord, SourceRef, array_digest
from .units import format_value

# fraction of pixels sitting exactly on the extreme value above which an image counts as saturated
SATURATION_FRACTION = 0.002


@dataclass(frozen=True)
class ScanFrame:
    """
    Where a scan sits on the sample: centre and size in metres and rotation in degrees.

    ``angle`` follows the Nanonis convention (positive = the scan frame turned clockwise as seen
    on screen).  Only used to place images and spectra relative to each other, never for lengths
    measured within one image.
    """

    center: tuple[float, float]
    size: tuple[float, float]
    angle: float = 0.0

    def pixel_to_world(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Pixel (x, y) of an image of ``shape`` (rows, cols) -> sample coordinates (m, y up)."""
        pts = np.asarray(points, float).reshape(-1, 2)
        rows, cols = shape
        u = (pts[:, 0] / cols - 0.5) * self.size[0]
        v = (0.5 - pts[:, 1] / rows) * self.size[1]
        a = math.radians(-self.angle)
        c, s = math.cos(a), math.sin(a)
        return np.column_stack([self.center[0] + c * u - s * v, self.center[1] + s * u + c * v])

    def world_to_pixel(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 2)
        rows, cols = shape
        dx, dy = pts[:, 0] - self.center[0], pts[:, 1] - self.center[1]
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        u, v = c * dx - s * dy, s * dx + c * dy
        return np.column_stack([(u / self.size[0] + 0.5) * cols, (0.5 - v / self.size[1]) * rows])

    def to_dict(self) -> dict[str, Any]:
        return {"center": list(self.center), "size": list(self.size), "angle": self.angle}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["ScanFrame"]:
        if not d:
            return None
        return cls(tuple(d["center"]), tuple(d["size"]), float(d.get("angle", 0.0)))  # type: ignore[arg-type]


@dataclass(frozen=True)
class QualityIssue:
    level: str        # "info" / "warning" / "error"
    code: str         # machine readable: "nan", "saturated", "anisotropic", "uncalibrated" ...
    message: str


@dataclass(frozen=True)
class ImageStats:
    """Statistics of the finite values (computed once per image)."""

    n: int
    n_invalid: int
    minimum: float
    maximum: float
    mean: float
    std: float
    p01: float
    p99: float

    @classmethod
    def of(cls, data: np.ndarray) -> "ImageStats":
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return cls(0, int(data.size), math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
        sample = finite if finite.size <= 2_000_000 else finite[:: finite.size // 2_000_000 + 1]
        p01, p99 = np.percentile(sample, [1, 99])
        return cls(int(finite.size), int(data.size - finite.size), float(finite.min()), float(finite.max()),
                   float(finite.mean()), float(finite.std()), float(p01), float(p99))


class Image:
    """A calibrated 2D measurement.  See the module docstring."""

    def __init__(self, data: Any, *, x: Optional[Axis] = None, y: Optional[Axis] = None, quantity: str = "",
                 unit: str = "", id: Optional[str] = None, source: Optional[SourceRef] = None,
                 metadata: Optional[dict] = None, history: Sequence[ProcessingRecord] = (),
                 frame: Optional[ScanFrame] = None, raw_dtype: Optional[str] = None, copy: bool = False):
        arr = np.array(data, dtype=np.float64, copy=True) if copy else np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(f"an Image is 2D, got shape {arr.shape}")
        if arr.dtype.kind not in "f":
            raw_dtype = raw_dtype or arr.dtype.str
            arr = arr.astype(np.float64)
        elif arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        if arr.flags.writeable:          # never share a buffer someone else can still modify
            if not copy:
                arr = arr.copy()
            arr.flags.writeable = False
        rows, cols = arr.shape
        x = x or Axis.pixels("x", cols)
        y = y or Axis.pixels("y", rows)
        if x.size != cols or y.size != rows:
            raise ValueError(f"axes ({x.size} x {y.size}) do not match the data ({cols} x {rows})")
        self._data = arr
        self.x, self.y = x, y
        self.quantity, self.unit = quantity, unit
        self.id = id or uuid.uuid4().hex[:12]
        self.source = source or SourceRef()
        self.metadata = dict(metadata or {})
        self.history: History = tuple(history)
        self.frame = frame
        self.raw_dtype = raw_dtype or arr.dtype.str

    # ---- data ---------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The values (read-only view; NaN = no data)."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape  # type: ignore[return-value]

    @cached_property
    def valid(self) -> np.ndarray:
        """Bool mask of finite samples."""
        return np.isfinite(self._data)

    @cached_property
    def stats(self) -> ImageStats:
        return ImageStats.of(self._data)

    @cached_property
    def digest(self) -> str:
        return array_digest(self._data)

    def filled(self, how: str = "nearest") -> np.ndarray:
        """A writable copy with invalid samples filled (``"nearest"`` valid neighbour or ``"median"``)."""
        out = np.array(self._data, dtype=np.float64)
        bad = ~self.valid
        if not bad.any():
            return out
        if bad.all():
            return np.zeros_like(out)
        if how == "median":
            out[bad] = float(np.median(out[~bad]))
            return out
        from scipy import ndimage
        _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
        return out[iy, ix]

    # ---- calibration ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.x.calibrated and self.y.calibrated and self.x.unit == self.y.unit

    @property
    def pixel_size(self) -> tuple[float, float]:
        """(dx, dy): physical size of one pixel along x and y (1, 1 when uncalibrated)."""
        if not self.calibrated:
            return 1.0, 1.0
        return abs(self.x.scale), abs(self.y.scale)

    @property
    def length_unit(self) -> str:
        return self.x.unit if self.calibrated else "px"

    @property
    def pixel_aspect(self) -> float:
        """dy / dx (1 for square pixels); the display stretches rows by this factor."""
        dx, dy = self.pixel_size
        return dy / dx

    @property
    def isotropic(self) -> bool:
        return abs(self.pixel_aspect - 1.0) < 1e-3

    @property
    def extent(self) -> tuple[float, float]:
        """(width, height) in the length unit."""
        dx, dy = self.pixel_size
        return self.shape[1] * dx, self.shape[0] * dy

    def to_physical(self, points: Any) -> np.ndarray:
        """Pixel (x, y) points -> physical (x, y) relative to the image's top-left corner."""
        pts = np.asarray(points, float).reshape(-1, 2)
        dx, dy = self.pixel_size
        return pts * np.array([dx, dy])

    def to_pixels(self, points: Any) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 2)
        dx, dy = self.pixel_size
        return pts / np.array([dx, dy])

    def calibration_text(self) -> str:
        if not self.calibrated:
            return "Uncalibrated: lengths in pixels."
        return f"x: {self.x.describe()}\ny: {self.y.describe()}"

    def sample(self, xs: Any, ys: Any, order: int = 1) -> np.ndarray:
        """Values at continuous pixel coordinates (bilinear by default, NaN outside / on invalid)."""
        from scipy import ndimage
        coords = np.vstack([np.asarray(ys, float).ravel() - 0.5, np.asarray(xs, float).ravel() - 0.5])
        vals = ndimage.map_coordinates(self.filled(), coords, order=order, mode="nearest")
        bad = ndimage.map_coordinates((~self.valid).astype(np.float32), coords, order=0, mode="constant", cval=1.0)
        vals[bad > 0.5] = np.nan
        return vals

    # ---- derived images -------------------------------------------------------------------
    def derive(self, data: Any, record: ProcessingRecord, *, x: Optional[Axis] = None, y: Optional[Axis] = None,
               quantity: Optional[str] = None, unit: Optional[str] = None, **changes: Any) -> "Image":
        """A new image with ``data`` and ``record`` appended to the history (same id unless changed)."""
        arr = np.asarray(data)
        if x is None and arr.shape[1] == self.shape[1]:
            x = self.x
        if y is None and arr.shape[0] == self.shape[0]:
            y = self.y
        return Image(arr, x=x, y=y, quantity=self.quantity if quantity is None else quantity,
                     unit=self.unit if unit is None else unit, id=changes.get("id", self.id),
                     source=changes.get("source", self.source),
                     metadata=changes.get("metadata", self.metadata), history=self.history + (record,),
                     frame=changes.get("frame", self.frame), raw_dtype=self.raw_dtype)

    def appended(self, *records: ProcessingRecord) -> "Image":
        """Same pixels with more history records (metadata-only operations)."""
        return Image(self._data, x=self.x, y=self.y, quantity=self.quantity, unit=self.unit, id=self.id,
                     source=self.source, metadata=self.metadata, history=self.history + tuple(records),
                     frame=self.frame, raw_dtype=self.raw_dtype)

    def with_calibration(self, dx: Optional[float], dy: Optional[float] = None, *,
                         calibration: Optional[Calibration] = None, unit: str = "m") -> "Image":
        """Same pixels, new pixel size (``None`` removes the calibration).  Recorded in the history."""
        cal = calibration or Calibration("manual")
        dy = dx if dy is None else dy
        x = self.x.with_calibration(dx, cal, unit)
        y = self.y.with_calibration(dy, cal, unit)
        rec = ProcessingRecord("calibration.set", {"dx": dx, "dy": dy, "unit": unit, **cal.to_dict()}, "none")
        img = Image(self._data, x=x, y=y, quantity=self.quantity, unit=self.unit, id=self.id, source=self.source,
                    metadata=self.metadata, history=self.history + (rec,), frame=self.frame, raw_dtype=self.raw_dtype)
        return img

    # ---- quality ------------------------------------------------------------------------------
    @cached_property
    def saturated(self) -> np.ndarray:
        """Bool mask of samples clipped at the data's extreme values (instrument limit reached)."""
        mask = np.zeros(self.shape, bool)
        st = self.stats
        if st.n < 100 or not math.isfinite(st.minimum) or st.maximum == st.minimum:
            return mask
        for v in (st.minimum, st.maximum):
            hit = self._data == v
            if hit.sum() >= max(10, SATURATION_FRACTION * st.n):
                mask |= hit
        if self.raw_dtype and np.dtype(self.raw_dtype).kind in "ui":
            info = np.iinfo(np.dtype(self.raw_dtype))
            for v in (info.min, info.max):
                hit = self._data == v
                if hit.sum() >= max(10, SATURATION_FRACTION * st.n):
                    mask |= hit
        return mask

    def quality(self) -> list[QualityIssue]:
        """Everything about this image that can make a measurement on it misleading."""
        issues: list[QualityIssue] = []
        st = self.stats
        total = self._data.size
        if st.n == 0:
            issues.append(QualityIssue("error", "empty", "The image contains no valid data."))
            return issues
        if st.n_invalid:
            valid = self.valid
            rows_bad = int((~valid).any(axis=1).sum())
            rows_empty = int((~valid.any(axis=1)).sum())
            frac = st.n_invalid / total
            pct = f"{frac:.1%}" if frac < 0.999 else f"{100 * frac:.2f}%"
            detail = f"{rows_bad} incomplete scan line(s)"
            if rows_empty:
                detail += f", {rows_empty} without any data"
            if st.n < 0.01 * total:
                detail += f"; only {st.n} pixel(s) were measured - the scan was probably stopped early"
            issues.append(QualityIssue("warning", "nan", f"{pct} of the pixels have no data ({detail}); "
                                                         "they are excluded from measurements."))
        n_sat = int(self.saturated.sum())
        if n_sat:
            issues.append(QualityIssue("warning", "saturated",
                                       f"{n_sat} pixel(s) ({n_sat / total:.2%}) sit exactly on the extreme value: "
                                       "the signal was probably clipped (heights there are lower bounds)."))
        if not self.calibrated:
            issues.append(QualityIssue("warning", "uncalibrated", "No lateral calibration: lengths are in pixels."))
        else:
            if not self.isotropic:
                dx, dy = self.pixel_size
                issues.append(QualityIssue("info", "anisotropic",
                                           f"Pixels are not square ({format_value(dx, self.x.unit)} × "
                                           f"{format_value(dy, self.y.unit)}); lengths and angles account for it."))
            for ax in (self.x, self.y):
                if ax.calibration.source in ("sidecar", "assumed"):
                    issues.append(QualityIssue("warning", "weak_calibration",
                                               f"{ax.name} scale {ax.calibration.describe()}."))
                    break
        if self.raw_dtype and np.dtype(self.raw_dtype).kind in "ui" and np.dtype(self.raw_dtype).itemsize == 1:
            issues.append(QualityIssue("warning", "8bit",
                                       "Values come from an 8-bit image: they are display levels, not physical "
                                       "heights, unless the file says otherwise."))
        if any(r.effect == "geometry" for r in self.history):
            issues.append(QualityIssue("info", "resampled", "The pixel grid was changed after loading."))
        return issues

    def describe(self) -> str:
        rows, cols = self.shape
        what = f"{self.quantity or 'value'} [{self.unit or 'a.u.'}]"
        size = f"{cols} × {rows} px"
        if self.calibrated:
            w, h = self.extent
            size += f" = {format_value(w, self.x.unit)} × {format_value(h, self.y.unit)}"
        return f"{what}, {size}, {self.source.describe()}"

    def __repr__(self) -> str:
        return f"Image({self.describe()}, {len(self.history)} op(s))"

    def header(self) -> dict[str, Any]:
        """Everything except the pixel values (for project files and exports)."""
        return {"id": self.id, "shape": list(self.shape), "quantity": self.quantity, "unit": self.unit,
                "x": self.x.to_dict(), "y": self.y.to_dict(), "source": self.source.to_dict(),
                "frame": self.frame.to_dict() if self.frame else None, "raw_dtype": self.raw_dtype,
                "history": [r.to_dict() for r in self.history]}


@dataclass
class ImageStack:
    """
    A sequence of images on the same grid (time series, bias series, a multi-page TIFF).
    ``frames`` is the stacking axis (e.g. ``Axis.from_values("bias", [...], "V")``).
    """

    images: list[Image]
    frames: Axis
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if len(self.images) != self.frames.size:
            raise ValueError("one image per frame expected")
        shapes = {im.shape for im in self.images}
        if len(shapes) > 1:
            raise ValueError(f"all frames of a stack need the same shape, got {sorted(shapes)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Image:
        return self.images[i]

    def __iter__(self) -> Iterator[Image]:
        return iter(self.images)

    def as_array(self) -> np.ndarray:
        return np.stack([im.data for im in self.images])


def replace_axes(img: Image, x: Axis, y: Axis) -> Image:
    return Image(img.data, x=x, y=y, quantity=img.quantity, unit=img.unit, id=img.id, source=img.source,
                 metadata=img.metadata, history=img.history, frame=img.frame, raw_dtype=img.raw_dtype)


__all__ = ["Image", "ImageStack", "ImageStats", "QualityIssue", "ScanFrame", "replace_axes", "replace"]
