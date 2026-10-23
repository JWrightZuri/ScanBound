"""
Opening data files as calibrated scientific data.

    from scanbound.io import open_data
    f = open_data("scan_0001.sxm")
    f.kind                      # "scan" (2D channels), "spectrum" (point spectroscopy) or "image"
    f.channels                  # [ChannelInfo(key="Z", quantity="Z", unit="m", direction="forward", ...)]
    img = f.image()             # default (topography) channel as an Image, in metres
    img = f.image("Current [bwd]")
    sp = open_data("spec_0001.dat").spectrum()

Every vendor reader of :mod:`scanbound.io.microscopy` is supported.  Lateral scales come back in
metres with their :class:`~scanbound.model.Calibration` (``file`` for instrument headers,
``sidecar`` for ScanBound's old ``metadata.csv``), the scan frame (offset, angle) when the file
has one, and NaN for missing samples.  Malformed files raise :class:`DataFormatError` with a
message meant for people.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..errors import ScanBoundError
from ..model import (DEFAULT_SCANNER_REL_UNCERTAINTY, Axis, Calibration, Image, ImageStack, ProcessingRecord,
                     ScanFrame, SourceRef, Spectrum)
from . import microscopy as mio
from .nanonis_dat import SpectrumFormatError, is_nanonis_dat, load_nanonis_dat

SPECTRUM_EXTENSIONS = (".dat",)
PLAIN_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")


class DataFormatError(ScanBoundError, ValueError):
    """A file could not be read as data (unsupported, truncated, corrupt ...)."""


@dataclass(frozen=True)
class ChannelInfo:
    key: str                 # unique within the file ("Z", "Z [bwd]")
    quantity: str            # "Z"
    unit: str
    direction: str = ""
    shape: tuple[int, int] = (0, 0)


@dataclass
class DataFile:
    """A file opened for reading; channels load lazily."""

    path: str
    format: str
    kind: str                                    # "scan" / "spectrum" / "image"
    channels: list[ChannelInfo] = field(default_factory=list)
    default_channel: str = ""
    metadata: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    header: dict = field(default_factory=dict)
    frame: Optional[ScanFrame] = None
    _mfile: Any = None
    _spectrum: Optional[Spectrum] = None
    _images: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    def channel_keys(self) -> list[str]:
        return [c.key for c in self.channels]

    def image(self, channel: Optional[str] = None) -> Image:
        if self.kind == "spectrum":
            raise DataFormatError(f"{self.name} is a spectrum, not an image")
        key = channel or self.default_channel
        if key not in self._images:
            self._images[key] = self._read_image(key)
        return self._images[key]

    def images(self) -> list[Image]:
        return [self.image(c.key) for c in self.channels]

    def spectrum(self) -> Spectrum:
        if self._spectrum is None:
            raise DataFormatError(f"{self.name} contains no spectrum")
        return self._spectrum

    # ---- reading -----------------------------------------------------------------------
    def _read_image(self, key: str) -> Image:
        mf = self._mfile
        ch = mf.channel(key)
        if ch is None:
            raise KeyError(f"{self.name}: no channel {key!r}; available: {', '.join(self.channel_keys())}")
        try:
            data = ch.data
        except (ValueError, OSError) as exc:
            raise DataFormatError(f"{self.name} / {key}: could not read the data ({exc})") from exc
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.size == 0:
            raise DataFormatError(f"{self.name} / {key}: not a 2D image (shape {data.shape})")
        rows, cols = data.shape
        cal = _channel_calibration(self.format, ch)
        x = Axis.spatial("x", cols, ch.px_size_x * 1e-9 if ch.px_size_x else None, calibration=cal)
        y = Axis.spatial("y", rows, ch.px_size_y * 1e-9 if ch.px_size_y else None, calibration=cal)
        quantity = key.split(" [")[0]
        rec = ProcessingRecord("io.read", {"format": self.format, "channel": key}, "none")
        data.flags.writeable = False
        return Image(data, x=x, y=y, quantity=quantity, unit=ch.unit or "",
                     id=f"{self.name}::{key}",
                     source=SourceRef.for_file(self.path, self.format, key, ch.direction or ""),
                     metadata={"file": dict(self.metadata)}, history=(rec,), frame=self.frame,
                     raw_dtype=getattr(ch, "raw_dtype", None))


def _channel_calibration(fmt: str, ch: Any) -> Calibration:
    if not (ch.px_size_x and ch.px_size_y):
        return Calibration()
    if getattr(ch, "manual_scale", False):
        return Calibration("manual", "pixel size entered by the user")
    if getattr(ch, "scale_source", "") == "sidecar":
        return Calibration("sidecar", "metadata.csv written by ScanBound's JPG export",
                           DEFAULT_SCANNER_REL_UNCERTAINTY)
    return Calibration("file", f"{fmt} header; scanner calibration uncertainty assumed "
                               f"{DEFAULT_SCANNER_REL_UNCERTAINTY:.0%}", DEFAULT_SCANNER_REL_UNCERTAINTY)


def _sxm_frame(header: dict) -> Optional[ScanFrame]:
    rng = mio._floats(header.get("SCAN_RANGE", ""))
    off = mio._floats(header.get("SCAN_OFFSET", ""))
    if len(rng) < 2:
        return None
    angle = (mio._floats(header.get("SCAN_ANGLE", "0")) or [0.0])[0]
    centre = (off[0], off[1]) if len(off) >= 2 else (0.0, 0.0)
    return ScanFrame(centre, (rng[0], rng[1]), angle)


def _plain_image(path: str) -> mio.MicroscopyFile:
    """8/16/32-bit and float images keeping their bit depth; RGB -> luminance; multi-page -> frames."""
    from PIL import Image as PILImage, ImageSequence

    try:
        pil = PILImage.open(path)
    except Exception as exc:  # noqa: BLE001 - PIL raises many types for corrupt files
        raise DataFormatError(f"{os.path.basename(path)}: not a readable image ({exc})") from exc
    channels: list = []
    with pil:
        mode = pil.mode
        for i, frame in enumerate(ImageSequence.Iterator(pil)):
            if i >= 512:
                break
            arr = np.asarray(frame)
            raw_dtype = arr.dtype.str
            if arr.ndim == 3:                         # RGB(A): luminance, as the old loader did
                arr = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
            elif arr.dtype == bool:
                arr = arr.astype(np.uint8)
                raw_dtype = "|b1"
            ch = mio.Channel("Intensity" if i == 0 else f"Intensity [{i}]", np.asarray(arr, dtype=np.float64), "")
            ch.raw_dtype = raw_dtype
            channels.append(ch)
        size = pil.size
    scale = mio._scale_from_metadata_csv(path)
    if scale:
        for ch in channels:
            ch.px_size_x, ch.px_size_y = scale
            ch.scale_source = "sidecar"
    md = OrderedDict([("Image mode", mode), ("Size", f"{size[0]} × {size[1]}"),
                      ("Frames", len(channels)),
                      ("Scale source", "metadata.csv" if scale else "none (set manually)")])
    return mio.MicroscopyFile(path, channels, md, OrderedDict(md), format="Image")


def _tiff_pages(path: str) -> int:
    from PIL import Image as PILImage
    try:
        with PILImage.open(path) as im:
            return int(getattr(im, "n_frames", 1))
    except Exception:  # noqa: BLE001 - the vendor loader reports unreadable files
        return 1


def open_data(path: "str | os.PathLike[str]") -> DataFile:
    """Open any supported file.  Raises :class:`DataFormatError` (or FileNotFoundError)."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        raise DataFormatError(f"{os.path.basename(path)} is empty (0 bytes)")
    low = path.lower()
    if low.endswith(SPECTRUM_EXTENSIONS):
        with open(path, "rb") as fh:
            head = fh.read(65536)
        if is_nanonis_dat(head):
            try:
                sp = load_nanonis_dat(path)
            except SpectrumFormatError as exc:
                raise DataFormatError(str(exc)) from exc
            md = OrderedDict(sp.metadata.get("summary", {}))
            return DataFile(path, "Nanonis spectroscopy", "spectrum", [], "", md, sp.metadata.get("header", {}),
                            _spectrum=sp)
    try:
        if low.endswith(PLAIN_IMAGE_EXTENSIONS) and (not low.endswith((".tif", ".tiff")) or _tiff_pages(path) > 1):
            mf = _plain_image(path)                   # multi-page TIFF: a stack of frames
        else:
            mf = mio.load_file(path)
            if mf.format == "Image":                 # TIFF fallback of the vendor loader
                mf = _plain_image(path)
    except DataFormatError:
        raise
    except (ValueError, KeyError, IndexError, OSError, EOFError) as exc:
        raise DataFormatError(f"{os.path.basename(path)}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - third-party parsers raise anything on corrupt files
        raise DataFormatError(f"{os.path.basename(path)}: unreadable ({type(exc).__name__}: {exc})") from exc
    if not mf.channels:
        raise DataFormatError(f"{os.path.basename(path)}: no 2D image found")

    infos = []
    for ch in mf.channels:
        try:
            shape = tuple(ch.shape) if ch._data is not None or ch._loader is None else (0, 0)
        except Exception:  # noqa: BLE001
            shape = (0, 0)
        infos.append(ChannelInfo(ch.name, ch.name.split(" [")[0], ch.unit or "", ch.direction or "", shape))  # type: ignore[arg-type]
    kind = "image" if mf.format in ("Image",) else "scan"
    frame = _sxm_frame(mf.header) if mf.format == "Nanonis" else None
    df = DataFile(path, mf.format, kind, infos, mf.default().name, OrderedDict(mf.metadata), dict(mf.header),
                  frame, _mfile=mf)
    if mf.format == "Nanonis":
        nx_ny = mio._floats(mf.header.get("SCAN_PIXELS", ""))
        if len(nx_ny) >= 2:
            shape = (int(nx_ny[1]), int(nx_ny[0]))
            df.channels = [ChannelInfo(c.key, c.quantity, c.unit, c.direction, shape) for c in infos]
        expected = _sxm_expected_size(mf)
        if expected and os.path.getsize(path) < expected:
            df.warnings.append(f"file is truncated ({os.path.getsize(path)} of {expected} bytes): "
                               "missing samples are shown as no data")
    return df


def _sxm_expected_size(mf: Any) -> Optional[int]:
    try:
        raw = open(mf.path, "rb").read(400_000)
        _, start = mio._sxm_header(raw)
        nx, ny = (int(v) for v in mio._floats(mf.header["SCAN_PIXELS"])[:2])
        kind = (mf.header.get("SCANIT_TYPE", "FLOAT").split() or ["FLOAT"])[0].upper()
        size = {"FLOAT": 4, "INT": 4, "UINT": 4, "DOUBLE": 8}.get(kind, 4)
        return start + len(mf.channels) * nx * ny * size
    except Exception:  # noqa: BLE001
        return None


# never data: skipped without opening them (a folder tree can hold many thousands)
NOT_DATA_EXTENSIONS = (".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log", ".py", ".pyc", ".ipynb", ".pdf",
                       ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".7z", ".gz", ".tar", ".exe",
                       ".dll", ".html", ".htm", ".xml", ".ini", ".cfg", ".bat", ".sh", ".pt", ".pkl", ".npy", ".npz",
                       ".h5", ".mp4", ".avi", ".mov", ".svg", ".lnk", ".db")


class ScanStopped(Exception):
    """A folder scan was cancelled."""


def is_data_file(path: str) -> bool:
    low = path.lower()
    if low.endswith(SPECTRUM_EXTENSIONS):
        return True
    if low.endswith(NOT_DATA_EXTENSIONS):
        return False
    return mio.format_for(path) is not None or low.endswith(PLAIN_IMAGE_EXTENSIONS)


def list_data_files(directory: str, recursive: bool = False, *, limit: Optional[int] = None,
                    progress: Optional[Callable[[int, int], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None) -> list[str]:
    """Data files in a folder, sorted by name (MATRIX channel files grouped as one scan).

    ``limit`` stops after that many files (the caller can tell by the length); ``progress(folders,
    files)`` is called after each folder; ``cancelled()`` returning True raises :class:`ScanStopped`."""
    out: list[str] = []
    n_dirs = 0
    for root, dirs, files in os.walk(directory):
        if cancelled is not None and cancelled():
            raise ScanStopped(directory)
        n_dirs += 1
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__"))) if recursive else []
        group_seen: set = set()
        for name in sorted(files, key=str.lower):
            path = os.path.join(root, name)
            if name.startswith(("~$", ".")) or not is_data_file(path):
                continue
            fmt = mio.format_for(path)
            if fmt is not None and fmt.name == "Omicron MATRIX":
                key = mio._mtrx_stem(path)
                if key in group_seen:
                    continue
                group_seen.add(key)
            out.append(path)
            if limit is not None and len(out) >= limit:
                return out
        if progress is not None:
            progress(n_dirs, len(out))
        if not recursive:
            break
    return out


def load_stack(path: str) -> ImageStack:
    """All frames of a multi-page image as an :class:`ImageStack` (frame axis = page index)."""
    f = open_data(path)
    imgs = [f.image(c.key) for c in f.channels]
    return ImageStack(imgs, Axis("frame", len(imgs), "", 1.0, 0.0, calibration=Calibration("file")))


def isfinite_or_none(v: Optional[float]) -> Optional[float]:
    return v if v is not None and math.isfinite(v) else None
