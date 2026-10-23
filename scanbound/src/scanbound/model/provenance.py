"""
Provenance: where data came from and what was done to it.

Every :class:`~scanbound.model.Image` carries a :class:`SourceRef` (the raw file, channel and
direction it was read from) and a tuple of :class:`ProcessingRecord` (each operation since then,
with its parameters).  Together they are enough to regenerate the image from the raw file, which
is what :meth:`scanbound.processing.ProcessingChain.regenerate` does.

``effect`` tells consumers what an operation did:

``values``     pixel values changed, the pixel grid did not (levelling, filters): annotations stay valid
``geometry``   the pixel grid changed (crop, resample, rotate); ``transform`` maps new -> old pixels
``derived``    a new kind of data was computed (a mask, an FFT): not comparable with the input
``none``       metadata only (a calibration change, a relabel)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

EFFECTS = ("values", "geometry", "derived", "none")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def software_version() -> str:
    from .. import __version__
    return f"scanbound {__version__}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def params_digest(params: Any) -> str:
    """Short stable hash of a parameter structure (JSON, sorted keys)."""
    text = json.dumps(_jsonable(params), sort_keys=True, default=str)
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def array_digest(data: np.ndarray) -> str:
    """Hash of an array's shape, dtype and bytes (NaN-safe; identical arrays give identical digests)."""
    arr = np.ascontiguousarray(data)
    h = hashlib.blake2b(digest_size=12)
    h.update(str((arr.shape, arr.dtype.str)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file (used to detect that a raw file changed after it was annotated)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class SourceRef:
    """The raw data an image or spectrum was read from."""

    path: str = ""
    format: str = ""
    channel: str = ""
    direction: str = ""          # "forward" / "backward" / ""
    frame: Optional[int] = None  # index in an image stack
    size: Optional[int] = None   # file size in bytes when read
    mtime: Optional[float] = None
    sha256: Optional[str] = None

    @classmethod
    def for_file(cls, path: str, fmt: str = "", channel: str = "", direction: str = "",
                 frame: Optional[int] = None) -> "SourceRef":
        try:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size = mtime = None
        return cls(os.path.abspath(path) if path else "", fmt, channel, direction, frame, size, mtime)

    def changed_on_disk(self) -> Optional[bool]:
        """True if the file's size or modification time differ from when it was read (None = unknown)."""
        if not self.path or self.size is None:
            return None
        try:
            st = os.stat(self.path)
        except OSError:
            return True
        return st.st_size != self.size or (self.mtime is not None and abs(st.st_mtime - self.mtime) > 1e-3)

    def describe(self) -> str:
        parts = [os.path.basename(self.path) or "(in memory)"]
        if self.channel:
            parts.append(self.channel + (f" ({self.direction})" if self.direction else ""))
        if self.frame is not None:
            parts.append(f"frame {self.frame}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SourceRef":
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass(frozen=True)
class ProcessingRecord:
    """One operation applied to data."""

    op: str                                   # e.g. "spm.plane", "calibration.manual", "detect.adapt"
    params: dict = field(default_factory=dict)
    effect: str = "values"
    note: str = ""
    software: str = ""
    time: str = ""
    transform: Optional[tuple[tuple[float, ...], ...]] = None   # 3x3 affine new px -> old px (geometry ops)

    def __post_init__(self) -> None:
        if self.effect not in EFFECTS:
            raise ValueError(f"unknown effect {self.effect!r}")
        object.__setattr__(self, "params", _jsonable(dict(self.params)))
        if not self.software:
            object.__setattr__(self, "software", software_version())
        if not self.time:
            object.__setattr__(self, "time", now_iso())

    @property
    def changes_pixels(self) -> bool:
        return self.effect in ("values", "geometry", "derived")

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items() if k != "shapes")
        return f"{self.op}({args})" + (f" - {self.note}" if self.note else "")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "params": self.params, "effect": self.effect,
                             "software": self.software, "time": self.time}
        if self.note:
            d["note"] = self.note
        if self.transform is not None:
            d["transform"] = [list(r) for r in self.transform]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingRecord":
        t = d.get("transform")
        return cls(str(d["op"]), dict(d.get("params") or {}), str(d.get("effect", "values")), str(d.get("note", "")),
                   str(d.get("software", "")), str(d.get("time", "")),
                   tuple(tuple(float(v) for v in r) for r in t) if t else None)


History = tuple[ProcessingRecord, ...]


def history_changes_geometry(history: History) -> bool:
    return any(r.effect == "geometry" for r in history)
