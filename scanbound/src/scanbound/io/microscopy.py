# mypy: ignore-errors
# (vendor file parsers carried over unchanged from core/microscopy_io.py)
"""
microscopy_io.py
================

Loads microscopy files of many vendors into one common structure, with the pixel scale
converted to nanometres and the image oriented for display (row 0 = top, column 0 = left).

    from scanbound.io import find_scans, load_file
    for path in find_scans(directory):
        f = load_file(path)
        ch = f.default()                       # topography / height channel if present
        ch.data, ch.unit, ch.px_size_x, ch.px_size_y     # px sizes in nm (None = uncalibrated)
        f.metadata                             # key settings, ready for display
        f.header                               # every header entry of the file

Supported formats
-----------------
Nanonis              .sxm
Omicron MATRIX       .Z_mtrx, .I_mtrx, ... (all channel files of one scan are grouped;
                     the session result file *_0001.mtrx must be in the same folder)
Asylum / Igor        .ibw
Bruker NanoScope     .spm and numbered files (.000, .001, ...) with a NanoScope header
JPK                  .jpk
Gwyddion             .gwy
WSxM                 .stp, .top
Gatan / FEI / TIFF   .dm3, .dm4, .emd, .ser, .tif, .tiff (calibration read by RosettaSciIO)
Plain images         .png, .jpg, .jpeg, .bmp  (scale taken from a ScanBound metadata.csv
                     in the same folder when present, otherwise uncalibrated)

Adding a format
---------------
Write a function ``path -> MicroscopyFile`` and register it::

    @register_format("RHK SM4", (".sm4",))
    def _load_sm4(path):
        ...
        return MicroscopyFile(path, channels=[Channel("Topography", data, "m", px_x_nm, px_y_nm)],
                              metadata={...}, header={...})

``sniff=`` (a function of the first 64 bytes) lets a loader claim files whose extension does
not identify them.  Use ``length_to_nm`` for unit conversion and orient the data so that
row 0 is the top of the image.
"""

from __future__ import annotations

import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# =============================================================================
# Units and formatting
# =============================================================================

_NM_PER_UNIT = {
    "m": 1e9, "meter": 1e9, "meters": 1e9, "metre": 1e9,
    "mm": 1e6, "um": 1e3, "µm": 1e3, "μm": 1e3, "~m": 1e3, "micron": 1e3, "microns": 1e3,
    "nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
    "å": 0.1, "Å": 0.1, "angstrom": 0.1, "angstroms": 0.1, "ang": 0.1, "a": 0.1,
    "pm": 1e-3,
}


def length_to_nm(value, unit) -> Optional[float]:
    """Length in nm, or None when `unit` is not a length unit (e.g. 1/nm, px, '')."""
    if value is None or unit is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    factor = _NM_PER_UNIT.get(str(unit).strip()) or _NM_PER_UNIT.get(str(unit).strip().lower())
    if factor is None or not math.isfinite(v):
        return None
    return v * factor


_SI_PREFIXES = [(1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, ""),
                (1e3, "k"), (1e6, "M"), (1e9, "G")]
_NO_PREFIX_UNITS = {"", "deg", "°", "%", "K", "C", "°C", "rad", "count", "counts", "arb", "a.u."}


def fmt_si(value, unit="", digits=4) -> str:
    """Number with an SI prefix: fmt_si(1e-10, 'A') -> '100 pA'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{value} {unit}".strip()
    if not math.isfinite(v) or v == 0 or unit in _NO_PREFIX_UNITS:
        return f"{v:.{digits}g} {unit}".strip()
    scale, prefix = _SI_PREFIXES[0]
    for s, p in _SI_PREFIXES:
        if abs(v) >= s * 0.9999:
            scale, prefix = s, p
    return f"{v / scale:.{digits}g} {prefix}{unit}".strip()


def fmt_length(nm, digits=4) -> str:
    """Length given in nm, shown in pm / nm / µm / mm as appropriate."""
    if nm is None:
        return "n/a"
    return fmt_si(nm * 1e-9, "m", digits)


# =============================================================================
# Data structures
# =============================================================================

class Channel:
    """
    One image of a file.  ``data`` is a 2D float array with row 0 = top of the image as it
    should be displayed and column 0 = left.  ``px_size_x`` / ``px_size_y`` are the physical
    size of one pixel in nm along the columns / rows (None when the file has no real-space
    calibration).  Data may be read lazily on first access.
    """

    def __init__(self, name, data=None, unit="", px_size_x=None, px_size_y=None, direction="",
                 loader: Optional[Callable[[], np.ndarray]] = None):
        self.name = str(name)
        self.unit = unit or ""
        self.px_size_x = px_size_x
        self.px_size_y = px_size_y
        self.direction = direction
        self._data = None if data is None else np.asarray(data, dtype=float)
        self._loader = loader
        self.manual_scale = False

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            if self._loader is None:
                raise ValueError(f"Channel {self.name!r} has no data")
            self._data = np.asarray(self._loader(), dtype=float)
            self._loader = None
        return self._data

    @property
    def shape(self):
        return self.data.shape

    @property
    def calibrated(self) -> bool:
        return bool(self.px_size_x) and bool(self.px_size_y)

    def scan_size(self):
        """(width, height) in nm of the full image, or None."""
        if not self.calibrated:
            return None
        rows, cols = self.shape
        return cols * self.px_size_x, rows * self.px_size_y

    def __repr__(self):
        return f"Channel({self.name!r}, unit={self.unit!r}, px=({self.px_size_x}, {self.px_size_y}) nm)"


@dataclass
class MicroscopyFile:
    path: str
    channels: list = field(default_factory=list)
    metadata: dict = field(default_factory=OrderedDict)   # key settings, display strings
    header: dict = field(default_factory=OrderedDict)     # full header, display strings
    format: str = ""
    default_channel: int = 0
    display_name: str = ""
    stem: str = ""                                        # base name for exported files

    def __post_init__(self):
        base = os.path.basename(self.path)
        self.display_name = self.display_name or base
        self.stem = self.stem or os.path.splitext(base)[0]

    def channel_names(self):
        return [c.name for c in self.channels]

    def channel(self, name) -> Optional[Channel]:
        for c in self.channels:
            if c.name == name:
                return c
        return None

    def default(self) -> Channel:
        return self.channels[self.default_channel]

    def set_manual_scale(self, px_nm_x, px_nm_y=None):
        """Override the pixel scale of every channel (nm per pixel)."""
        for c in self.channels:
            c.px_size_x = float(px_nm_x)
            c.px_size_y = float(px_nm_y if px_nm_y is not None else px_nm_x)
            c.manual_scale = True


# =============================================================================
# Format registry
# =============================================================================

@dataclass
class FileFormat:
    name: str
    extensions: tuple
    loader: Callable
    sniff: Optional[Callable] = None
    match: Optional[Callable] = None     # custom path matcher (overrides extensions)


FORMATS: list = []


def register_format(name, extensions, sniff=None, match=None):
    def deco(func):
        FORMATS.append(FileFormat(name, tuple(e.lower() for e in extensions), func, sniff, match))
        return func
    return deco


def supported_extensions():
    return sorted({e for f in FORMATS for e in f.extensions})


def file_dialog_filter():
    pats = " ".join(f"*{e}" for e in supported_extensions())
    lines = [f"Microscopy files ({pats})"]
    lines += [f"{f.name} ({' '.join('*' + e for e in f.extensions)})" for f in FORMATS]
    return ";;".join(lines + ["All files (*)"])


def format_for(path) -> Optional[FileFormat]:
    low = path.lower()
    for fmt in FORMATS:
        if fmt.match is not None:
            if fmt.match(path):
                return fmt
        elif low.endswith(fmt.extensions):
            return fmt
    sniffers = [f for f in FORMATS if f.sniff is not None]
    if sniffers and os.path.isfile(path):
        try:
            with open(path, "rb") as fh:
                head = fh.read(64)
        except OSError:
            return None
        for fmt in sniffers:
            if fmt.sniff(head):
                return fmt
    return None


_FOREIGN_MAGIC = [(b"\x89PNG", "a PNG image"), (b"\xff\xd8\xff", "a JPEG image"), (b"BM", "a BMP image"),
                  (b"PK\x03\x04", "a ZIP archive"), (b"%PDF", "a PDF document")]
_IMAGE_FORMATS = ("Image", "Gatan / FEI / TIFF")


def load_file(path, fmt: Optional[FileFormat] = None) -> MicroscopyFile:
    """Load a file with the matching loader. Raises ValueError with a readable reason on failure."""
    fmt = fmt or format_for(path)
    if fmt is None:
        raise ValueError(f"Unsupported file type: {os.path.basename(path)}")
    if fmt.name not in _IMAGE_FORMATS:
        with open(path, "rb") as fh:
            head = fh.read(8)
        for magic, what in _FOREIGN_MAGIC:
            if head.startswith(magic):
                raise ValueError(f"{os.path.basename(path)} is actually {what}, not a {fmt.name} file")
    result = fmt.loader(path)
    result.format = result.format or fmt.name
    if not result.channels:
        raise ValueError(f"No 2D image found in {os.path.basename(path)}")
    result.default_channel = pick_default_channel(result.channels)
    return result


def find_scans(directory) -> list:
    """Paths of all loadable scans in a directory (one entry per MATRIX scan), sorted by name."""
    paths, seen_groups = [], set()
    for name in sorted(os.listdir(directory), key=str.lower):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        fmt = format_for(path)
        if fmt is None:
            continue
        if fmt.name == "Omicron MATRIX":
            key = _mtrx_stem(path)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            path = _mtrx_group(path)[0]
        paths.append(path)
    return paths


_TOPO_WORDS = ("height sensor", "zsensor", "height", "topograph", "z")


def pick_default_channel(channels) -> int:
    """Index of the topography channel (forward / trace preferred), else 0."""
    def score(ch):
        name = ch.name.lower()
        base = re.split(r"[\s(\[]", name, maxsplit=1)[0]
        s = 0
        for rank, word in enumerate(_TOPO_WORDS):
            if name.startswith(word) and (word != "z" or base == "z"):
                s = 10 - rank
                break
        if any(w in name for w in ("backward", "bwd", "retrace", "backward/")):
            s -= 0.5
        return s
    best = max(range(len(channels)), key=lambda i: (score(channels[i]), -i))
    return best if score(channels[best]) > 0 else 0


# =============================================================================
# Nanonis SXM
# =============================================================================

def _sxm_header(raw: bytes):
    end = raw.index(b":SCANIT_END:")
    header = OrderedDict()
    key = None
    for line in raw[:end].decode("latin-1").split("\n"):
        line = line.rstrip("\r")
        if len(line) > 1 and line.startswith(":") and line.endswith(":"):
            key = line[1:-1]
            header[key] = []
        elif key is not None and line.strip():
            header[key].append(line)
    data_start = raw.index(b"\x1a\x04", end) + 2
    return OrderedDict((k, "\n".join(v).strip()) for k, v in header.items()), data_start


def _table(text):
    rows = [[c.strip() for c in line.strip("\t ").split("\t")] for line in text.split("\n") if line.strip()]
    if len(rows) < 2:
        return []
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def _floats(text):
    return [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text or "")]


@register_format("Nanonis", (".sxm",))
def _load_sxm(path):
    with open(path, "rb") as fh:
        raw = fh.read(200_000)
    if not raw.startswith(b":NANONIS_VERSION:"):
        raise ValueError("missing Nanonis header")
    if b":SCANIT_END:" not in raw:                  # very long header
        with open(path, "rb") as fh:
            raw = fh.read()
    header, data_start = _sxm_header(raw)

    nx, ny = (int(v) for v in _floats(header["SCAN_PIXELS"])[:2])
    range_m = _floats(header.get("SCAN_RANGE", ""))
    px_x = range_m[0] / nx * 1e9 if len(range_m) >= 2 else None
    px_y = range_m[1] / ny * 1e9 if len(range_m) >= 2 else None
    kind, order = (header.get("SCANIT_TYPE", "FLOAT MSBFIRST").split() + ["MSBFIRST"])[:2]
    dtype = np.dtype({"FLOAT": "f4", "INT": "i4", "UINT": "u4", "DOUBLE": "f8"}[kind.upper()]) \
        .newbyteorder(">" if order.upper() == "MSBFIRST" else "<")
    scan_up = header.get("SCAN_DIR", "").strip().lower() == "up"

    channels, index = [], 0
    for row in _table(header["DATA_INFO"]):
        direction = row.get("Direction", "forward").lower()
        for d in (("forward", "backward") if direction == "both" else (direction,)):
            offset = data_start + index * nx * ny * dtype.itemsize
            index += 1

            def read(offset=offset, d=d):
                flat = np.fromfile(path, dtype=dtype, count=nx * ny, offset=offset)
                if flat.size < nx * ny:                # truncated file: keep what is there, rest = no data
                    flat = np.concatenate([flat.astype(float), np.full(nx * ny - flat.size, np.nan)])
                data = flat.reshape(ny, nx)
                if d == "backward":
                    data = data[:, ::-1]           # stored in scan order (right to left)
                if scan_up:
                    data = data[::-1, :]           # first line stored is the bottom one
                return data.astype(float)

            name = row["Name"] if d == "forward" else f"{row['Name']} [bwd]"
            channels.append(Channel(name, unit=row.get("Unit", ""), px_size_x=px_x, px_size_y=px_y,
                                    direction=d, loader=read))

    md = OrderedDict()
    md["Acquired"] = f"{header.get('REC_DATE', '').strip()} {header.get('REC_TIME', '').strip()}".strip()
    if len(range_m) >= 2:
        md["Scan range"] = f"{fmt_length(range_m[0] * 1e9)} × {fmt_length(range_m[1] * 1e9)}"
    offset_m = _floats(header.get("SCAN_OFFSET", ""))
    if len(offset_m) >= 2:
        md["Scan offset"] = f"{fmt_length(offset_m[0] * 1e9)}, {fmt_length(offset_m[1] * 1e9)}"
    md["Scan angle"] = f"{(_floats(header.get('SCAN_ANGLE', '0')) or [0.0])[0]:g} °"
    md["Scan direction"] = header.get("SCAN_DIR", "")
    if "BIAS" in header:
        md["Bias"] = fmt_si(_floats(header["BIAS"])[0], "V")
    if "Z-Controller>Setpoint" in header:
        md["Setpoint"] = fmt_si(_floats(header["Z-Controller>Setpoint"])[0],
                                header.get("Z-Controller>Setpoint unit", "").strip())
    else:
        zc = _table(header.get("Z-CONTROLLER", ""))
        if zc and "Setpoint" in zc[0]:
            md["Setpoint"] = zc[0]["Setpoint"]
    if "Z-Controller>Controller name" in header:
        md["Feedback"] = f"{header['Z-Controller>Controller name']} " \
                         f"({header.get('Z-Controller>Controller status', '').strip()})"
    line_time = _floats(header.get("SCAN_TIME", ""))
    if line_time:
        md["Line time (fwd)"] = fmt_si(line_time[0], "s")
    if "Scan>speed forw. (m/s)" in header:
        md["Tip speed"] = fmt_si(_floats(header["Scan>speed forw. (m/s)"])[0], "m/s")
    if "ACQ_TIME" in header:
        md["Acquisition time"] = fmt_si(_floats(header["ACQ_TIME"])[0], "s")
    if "REC_TEMP" in header:
        md["Temperature"] = f"{_floats(header['REC_TEMP'])[0]:.4g} K"
    if header.get("Lock-in>Lock-in status", "").strip().upper() == "ON":
        md["Lock-in"] = (f"{fmt_si(_floats(header.get('Lock-in>Amplitude', '0'))[0], 'V')} @ "
                         f"{fmt_si(_floats(header.get('Lock-in>Frequency (Hz)', '0'))[0], 'Hz')}")
    if header.get("COMMENT"):
        md["Comment"] = header["COMMENT"].replace("\n", "; ")
    if "NanonisMain>SW Version" in header:
        md["Software"] = f"Nanonis {header['NanonisMain>SW Version']}"
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# Omicron MATRIX
# =============================================================================

def _is_mtrx_image(path):
    name = os.path.basename(path)
    if not name.lower().endswith("_mtrx") or "." not in name:
        return False
    ext = name.rsplit(".", 1)[1]
    return "(" not in ext                           # I(V)_mtrx, Z(t)_mtrx ... are curves


def _mtrx_stem(path):
    return os.path.join(os.path.dirname(path), os.path.basename(path).rsplit(".", 1)[0])


def _mtrx_group(path):
    """All image channel files of the scan `path` belongs to (Z first)."""
    stem = _mtrx_stem(path)
    directory, base = os.path.split(stem)
    files = [os.path.join(directory, n) for n in os.listdir(directory or ".")
             if n.startswith(base + ".") and _is_mtrx_image(os.path.join(directory, n))]
    return sorted(files, key=lambda p: (not p.endswith(".Z_mtrx"), p.lower()))


@register_format("Omicron MATRIX", ("_mtrx",), match=_is_mtrx_image)
def _load_mtrx(path):
    import access2thematrix

    channels, header, md = [], OrderedDict(), OrderedDict()
    param = None
    for fp in _mtrx_group(path):
        m = access2thematrix.MtrxData()
        traces, message = m.open(fp)
        if not traces:
            raise ValueError(f"{os.path.basename(fp)}: {message}")
        chan = os.path.basename(fp).rsplit(".", 1)[1][:-len("_mtrx")]
        for trace in traces.values():
            im, _ = m.select_image(trace)
            data = getattr(im, "data", None)
            if data is None or data.ndim != 2 or data.size == 0:
                continue
            rows, cols = data.shape
            # access2thematrix puts the bottom line in row 0 for every trace.
            channels.append(Channel(f"{chan} {trace}", np.flipud(data), im.channel_name_and_unit[1],
                                    px_size_x=im.width / cols * 1e9, px_size_y=im.height / rows * 1e9,
                                    direction=str(trace)))
        if param is None:
            param = m.param

    p = param or {}

    def val(key):
        v = p.get(key)
        return v[0] if isinstance(v, (list, tuple)) else v

    def unit(key):
        v = p.get(key)
        return v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else ""

    md["Acquired"] = str(val("BKLT") or "")
    md["Sample"] = str(val("MARK::MTRX.SAMPLE_NAME") or "")
    md["Experiment"] = str(val("EXPD::s0") or "")
    if val("EEPA::XYScanner.Width") is not None:
        md["Scan range"] = f"{fmt_length(val('EEPA::XYScanner.Width') * 1e9)} × " \
                           f"{fmt_length(val('EEPA::XYScanner.Height') * 1e9)}"
        md["Points × lines"] = f"{val('EEPA::XYScanner.Points')} × {val('EEPA::XYScanner.Lines')}"
        md["Scan offset"] = f"{fmt_length(val('EEPA::XYScanner.X_Offset') * 1e9)}, " \
                            f"{fmt_length(val('EEPA::XYScanner.Y_Offset') * 1e9)}"
        md["Scan angle"] = f"{val('EEPA::XYScanner.Angle')} °"
        md["Raster time"] = fmt_si(val("EEPA::XYScanner.Raster_Time"), "s")
    if val("EEPA::GapVoltageControl.Voltage") is not None:
        md["Bias"] = fmt_si(val("EEPA::GapVoltageControl.Voltage"), "V")
    if val("EEPA::Regulator.Setpoint_1") is not None:
        u = {"Ampere": "A", "Volt": "V", "Meter": "m"}.get(unit("EEPA::Regulator.Setpoint_1"),
                                                           unit("EEPA::Regulator.Setpoint_1"))
        md["Setpoint"] = fmt_si(val("EEPA::Regulator.Setpoint_1"), u)
        md["Loop gain I"] = f"{val('EEPA::Regulator.Loop_Gain_1_I')} %"
        md["Feedback"] = "on" if val("EEPA::Regulator.Feedback_Loop_Enabled") else "off"
    for k, v in p.items():
        header[k] = " ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
    stem = os.path.basename(_mtrx_stem(path))
    names = ", ".join(os.path.basename(f).rsplit(".", 1)[1][:-5] for f in _mtrx_group(path))
    return MicroscopyFile(path, channels, OrderedDict((k, v) for k, v in md.items() if v), header,
                          display_name=f"{stem} [{names}]", stem=stem)


# =============================================================================
# Igor binary wave (Asylum Research)
# =============================================================================

_IBW_TYPES = {2: "f4", 4: "f8", 8: "i1", 0x10: "i2", 0x20: "i4", 0x48: "u1", 0x50: "u2", 0x60: "u4"}


def _cstr(b):
    return b.split(b"\x00", 1)[0].decode("latin-1")


@register_format("Igor / Asylum", (".ibw",))
def _load_ibw(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    endian = "<"
    version = int.from_bytes(raw[:2], "little")
    if version not in (5,):
        if int.from_bytes(raw[:2], "big") == 5:
            endian, version = ">", 5
        else:
            raise ValueError(f"Igor binary wave version {version} is not supported (only v5)")
    i4 = np.dtype(endian + "i4")
    (wfm_size, formula_size, note_size, data_e_units_size) = np.frombuffer(raw, i4, 4, 4)
    dim_e_units_size = np.frombuffer(raw, i4, 4, 20)
    dim_labels_size = np.frombuffer(raw, i4, 4, 36)
    wh = 64
    npnts = int(np.frombuffer(raw, i4, 1, wh + 12)[0])
    wtype = int(np.frombuffer(raw, np.dtype(endian + "i2"), 1, wh + 16)[0])
    n_dim = [int(v) for v in np.frombuffer(raw, i4, 4, wh + 68)]
    sf_a = np.frombuffer(raw, np.dtype(endian + "f8"), 4, wh + 84)
    data_units = _cstr(raw[wh + 148:wh + 152])
    dim_units = [_cstr(raw[wh + 152 + 4 * i:wh + 156 + 4 * i]) for i in range(4)]
    if wtype & 1:
        raise ValueError("complex waves are not supported")
    dtype = np.dtype(endian + _IBW_TYPES[wtype])
    data_off = 64 + 320
    data = np.frombuffer(raw, dtype, npnts, data_off)

    pos = data_off + wfm_size - 320
    pos += formula_size
    note = raw[pos:pos + note_size].decode("latin-1")
    pos += note_size
    data_e_units = raw[pos:pos + data_e_units_size].decode("latin-1")
    pos += data_e_units_size
    ext_dim_units = []
    for size in dim_e_units_size:
        ext_dim_units.append(raw[pos:pos + size].decode("latin-1"))
        pos += size
    labels = []
    for d, size in enumerate(dim_labels_size):
        block = raw[pos:pos + size]
        labels.append([_cstr(block[k:k + 32]) for k in range(0, len(block), 32)])
        pos += size
    if n_dim[1] == 0:
        raise ValueError("1D wave, not an image")

    dims = [n for n in n_dim if n > 0]
    # Igor is column-major: point (p, q, r) is at p + nP * (q + nQ * r)
    cube = data.reshape(dims[::-1])                 # [layer][q][p]
    if cube.ndim == 2:
        cube = cube[None]
    elif cube.ndim == 4:
        cube = cube.reshape(-1, cube.shape[-2], cube.shape[-1])

    units_x = ext_dim_units[0] or dim_units[0]
    units_y = ext_dim_units[1] or dim_units[1]
    px_x = length_to_nm(abs(sf_a[0]), units_x)
    px_y = length_to_nm(abs(sf_a[1]), units_y)
    layer_labels = labels[2][1:] if len(labels) > 2 and len(labels[2]) > 1 else []

    notes = OrderedDict()
    for line in re.split(r"[\r\n]+", note):
        k, sep, v = line.partition(":")
        if sep and k.strip():
            notes[k.strip()] = v.strip()

    channels = []
    for i, layer in enumerate(cube):
        name = layer_labels[i] if i < len(layer_labels) and layer_labels[i] else f"Layer {i}"
        low = name.lower()
        unit = (data_e_units or data_units) if not layer_labels else \
            ("deg" if "phase" in low else "m" if any(w in low for w in ("height", "zsensor", "amplitude",
                                                                         "deflection", "z ")) else "")
        # rows = q (slow axis, y up) -> flip so that row 0 is the top
        channels.append(Channel(name, np.flipud(layer.astype(float)), unit, px_x, px_y))

    md = OrderedDict()
    for key, label in (("Date", "Date"), ("Time", "Time"), ("ImagingMode", "Imaging mode"),
                       ("MicroscopeModel", "Microscope"), ("ScanSize", "Scan size"),
                       ("FastScanSize", "Fast scan size"), ("SlowScanSize", "Slow scan size"),
                       ("ScanPoints", "Points"), ("ScanLines", "Lines"), ("ScanRate", "Scan rate (Hz)"),
                       ("ScanAngle", "Scan angle (°)"), ("SetPoint", "Setpoint"),
                       ("AmplitudeSetpointVolts", "Amplitude setpoint (V)"), ("DriveAmplitude", "Drive amplitude"),
                       ("DriveFrequency", "Drive frequency (Hz)"), ("IntegralGain", "Integral gain"),
                       ("SpringConstant", "Spring constant (N/m)"), ("InvOLS", "InvOLS"),
                       ("SurfaceVoltage", "Surface voltage (V)"), ("TipVoltage", "Tip voltage (V)"),
                       ("ImageNote", "Note")):
        if notes.get(key):
            v = notes[key]
            if key in ("ScanSize", "FastScanSize", "SlowScanSize"):
                v = fmt_length(float(v) * 1e9) if _floats(v) else v
            md[label] = v
    header = OrderedDict([("Wave name", _cstr(raw[wh + 28:wh + 60])), ("Dimensions", " × ".join(map(str, dims))),
                          ("Scale (dim 0)", f"{sf_a[0]} {units_x}"), ("Scale (dim 1)", f"{sf_a[1]} {units_y}")])
    header.update(notes)
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# Bruker NanoScope
# =============================================================================

def _nanoscope_sniff(head: bytes):
    return head.lstrip(b"?\\").startswith(b"*File list") or head.startswith(b"\\*EC File list")


def _nanoscope_header(raw: bytes):
    end = raw.find(b"\\*File list end")
    text = raw[:end if end >= 0 else len(raw)].decode("latin-1")
    sections = []
    for line in re.split(r"\r?\n", text):
        line = line.strip()
        if not line.startswith("\\"):
            continue
        line = line[1:]
        if line.startswith("*"):
            sections.append((line[1:], OrderedDict()))
            continue
        if not sections:
            sections.append(("File list", OrderedDict()))
        key, sep, value = line.partition(": ")
        if not sep:
            key, value = line.rstrip(":"), ""
        sections[-1][1][key.strip()] = value.strip()
    return sections


def _scan_size(text):
    """'505.859 505.859 nm' -> (505.859, 505.859, 'nm'); '5000 nm' -> (5000, None, 'nm')."""
    nums = _floats(text)
    unit = text.split()[-1] if text.split() else ""
    if not nums:
        return None
    return nums[0], (nums[1] if len(nums) > 1 else None), unit


@register_format("Bruker NanoScope", (".spm",), sniff=_nanoscope_sniff)
def _load_nanoscope(path):
    with open(path, "rb") as fh:
        head = fh.read(400_000)
    if not _nanoscope_sniff(head[:64]):
        raise ValueError("missing NanoScope header")
    sections = _nanoscope_header(head)
    general = OrderedDict()
    for name, sec in sections:
        if name != "Ciao image list":
            for k, v in sec.items():
                general.setdefault(k, v)

    def sens(name):
        """'V 4.273442 nm/V' -> (4.273442, 'nm')."""
        v = general.get(f"@{name}") or general.get(name)
        if not v:
            return None
        m = re.match(r"\s*[A-Z]\s+([-+0-9.eE]+)\s*(\S*)", v)
        if not m:
            return None
        return float(m.group(1)), m.group(2).split("/")[0]

    channels, used = [], {}
    for name, sec in sections:
        if name != "Ciao image list":
            continue
        cols = int(_floats(sec.get("Samps/line", "0"))[0])
        rows = int(_floats(sec.get("Number of lines", "0"))[0])
        offset = int(_floats(sec.get("Data offset", "0"))[0])
        length = int(_floats(sec.get("Data length", "0"))[0])
        if not cols or not rows:
            continue
        header_bpp = int(_floats(sec.get("Bytes/pixel", "0") or "0")[0]) or None
        bpp = length // (cols * rows) if length else (header_bpp or 2)
        bpp = bpp if bpp in (2, 4, 8) else 2
        # The Z scale counts in units of the *declared* bytes/pixel (newer files store 16-bit
        # channels as 4-byte integers but keep "Bytes/pixel: 2").
        scale_bpp = header_bpp or bpp
        title_src = sec.get("@2:Image Data") or sec.get("Image data") or ""
        m = re.search(r'"([^"]*)"', title_src)
        title = m.group(1) if m else (title_src.split("[")[-1].rstrip("]") or f"Image {len(channels)}")
        direction = sec.get("Line Direction", "").strip()
        label = f"{title} ({direction})" if direction else title
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label} #{used[label]}"

        # Z scale: 'V [Sens. Zsens] (0.001952197 V/LSB) 127.9392 V' -> value per count = 127.9392 / 256**bytes
        zs = sec.get("@2:Z scale") or sec.get("Z scale") or ""
        factor, unit = 1.0, ""
        m_val = re.search(r"\)\s*([-+0-9.eE]+)\s*(\S*)\s*$", zs)
        if m_val:
            factor, unit = float(m_val.group(1)) / 256 ** scale_bpp, m_val.group(2)
        else:
            m_hard = re.search(r"\(([-+0-9.eE]+)\s*([^)/]*)/LSB\)", zs)
            if m_hard:
                factor, unit = float(m_hard.group(1)), m_hard.group(2)
        m_sens = re.search(r"\[([^\]]+)\]", zs)
        if m_sens:
            s = sens(m_sens.group(1))
            if s is not None:
                factor *= s[0]
                unit = s[1] or unit

        size = _scan_size(sec.get("Scan Size") or sec.get("Scan size") or
                          general.get("Scan Size") or general.get("Scan size") or "")
        px_x = px_y = None
        if size is not None:
            x, y, u = size
            if y is None:
                ratio = _floats(sec.get("Aspect Ratio") or general.get("Aspect Ratio") or "1:1")
                y = x * (ratio[1] / ratio[0]) if len(ratio) == 2 and ratio[0] else x
            px_x = length_to_nm(x / cols, u)
            px_y = length_to_nm(y / rows, u)
        valid_rows = int(_floats(sec.get("Valid data len Y", str(rows)))[0] or rows)

        def read(offset=offset, bpp=bpp, cols=cols, rows=rows, factor=factor, valid_rows=valid_rows):
            dtype = {2: "<i2", 4: "<i4", 8: "<i8"}[bpp]
            data = np.fromfile(path, dtype=dtype, count=cols * rows, offset=offset).reshape(rows, cols)
            data = data[:min(valid_rows, rows)].astype(float) * factor
            return data[::-1, :]                   # first line stored is the bottom one

        channels.append(Channel(label, unit=unit, px_size_x=px_x, px_size_y=px_y, direction=direction,
                                loader=read))

    md = OrderedDict()
    for key, label in (("Date", "Date"), ("Version", "Version"), ("Operating mode", "Operating mode"),
                       ("@MicroscopeList", "Microscope / mode"), ("Scan Size", "Scan size"),
                       ("Samps/line", "Samples / line"), ("Aspect Ratio", "Aspect ratio"),
                       ("Scan Rate", "Scan rate (Hz)"), ("Rotate Ang.", "Rotation"), ("Capture direction", "Capture direction"),
                       ("Tip Velocity", "Tip velocity"), ("Setpoint Units", "Setpoint units")):
        if general.get(key):
            v = general[key]
            if key.startswith("@"):
                m = re.search(r'"([^"]*)"', v)
                v = m.group(1) if m else v
            md[label] = v
    header = OrderedDict()
    for name, sec in sections:
        for k, v in sec.items():
            header[f"{name} › {k}"] = v
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# JPK (TIFF with custom tags)
# =============================================================================

_JPK_SLOT_STRIDE = 0x30


@register_format("JPK", (".jpk",))
def _load_jpk(path):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        pages = list(tif.pages)
        meta = {t.code: t.value for t in pages[0].tags.values()}
        if 0x8042 not in meta or 0x8046 not in meta:
            raise ValueError("JPK grid tags not found")
        px_x = meta[0x8042] / meta[0x8046] * 1e9          # Grid_uLength [m] / Grid_iLength [px]
        px_y = meta[0x8043] / meta[0x8047] * 1e9          # Grid_vLength [m] / Grid_jLength [px]

        channels = []
        for page in pages[1:]:
            tags = {t.code: t.value for t in page.tags.values()}
            if 0x8050 not in tags:
                continue
            raw = page.asarray().astype(float)
            default = tags.get(0x8081)
            n_slots = int(tags.get(0x8080, 0))
            factor, offset, unit = 1.0, 0.0, ""
            for i in range(n_slots):
                if tags.get(0x8090 + i * _JPK_SLOT_STRIDE) == default:
                    base = 0x80A0 + i * _JPK_SLOT_STRIDE
                    unit = str(tags.get(base + 2, ""))
                    if tags.get(base + 3) == "LinearScaling":
                        factor = float(tags.get(base + 4, 1.0))
                        offset = float(tags.get(base + 5, 0.0))
                    break
            info = str(tags.get(0x8053, ""))
            retrace = re.search(r"retrace\s*:\s*true", info) is not None
            name = f"{tags.get(0x8052) or tags.get(0x8050)} ({'retrace' if retrace else 'trace'})"
            # JPK stores the first (bottom) line first
            channels.append(Channel(name, np.flipud(raw * factor + offset), unit, px_x, px_y,
                                    direction="retrace" if retrace else "trace"))

    md = OrderedDict()
    for code, label in ((0x8003, "Start"), (0x8006, "End"), (0x8004, "Name"), (0x8005, "Comment"),
                        (0x8009, "Account"), (0x8030, "Feedback mode")):
        v = meta.get(code)
        if v not in (None, "", "<empty>"):
            md[label] = str(v)
    md["Scan size"] = f"{fmt_length(meta[0x8042] * 1e9)} × {fmt_length(meta[0x8043] * 1e9)}"
    md["Pixels"] = f"{meta[0x8046]} × {meta[0x8047]}"
    if 0x8044 in meta:
        md["Grid angle"] = f"{math.degrees(float(meta[0x8044])):.4g} °"
    header = OrderedDict()
    for code in sorted(meta):
        if code < 0x8000:
            continue
        value = meta[code]
        if isinstance(value, str) and " : " in value and "\n" in value:
            for line in value.split("\n"):
                k, sep, v = line.partition(" : ")
                if sep:
                    header[k.strip()] = v.strip().replace("\\:", ":")
                    if re.search(r"(setpoint|line-rate|scan-rate|feedback-mode|i-gain|p-gain)$", k.strip()):
                        md[k.strip()] = v.strip()
        else:
            header[f"Tag 0x{code:04x}"] = str(value)
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# Gwyddion
# =============================================================================

@register_format("Gwyddion", (".gwy",))
def _load_gwy(path):
    import gwyfile

    obj = gwyfile.load(path)
    ids = sorted({int(m.group(1)) for k in obj.keys() for m in [re.match(r"^/(\d+)/data$", k)] if m})
    channels, header = [], OrderedDict()
    for i in ids:
        df = obj[f"/{i}/data"]
        title = obj.get(f"/{i}/data/title", f"Channel {i}")
        xy_unit = (df.get("si_unit_xy") or {}).get("unitstr", "m")
        z_unit = (df.get("si_unit_z") or {}).get("unitstr", "")
        data = np.asarray(df.data, dtype=float)                 # Gwyddion: row 0 is the top
        rows, cols = data.shape
        channels.append(Channel(title, data, z_unit, length_to_nm(df["xreal"] / cols, xy_unit),
                                length_to_nm(df["yreal"] / rows, xy_unit)))
        meta = obj.get(f"/{i}/meta")
        if meta:
            for k, v in meta.items():
                if k not in ("__type__",):
                    header.setdefault(str(k), str(v))
    md = OrderedDict((k, header[k]) for k in list(header)[:25])
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# WSxM
# =============================================================================

@register_format("WSxM", (".stp", ".top"))
def _load_wsxm(path):
    with open(path, "rb") as fh:
        start = fh.read(200)
        m = re.search(rb"Image header size: (\d+)", start)
        if not m:
            raise ValueError("WSxM header size not found")
        fh.seek(0)
        size = int(m.group(1))
        text = fh.read(size).decode("latin-1")
    header = OrderedDict()
    section = ""
    for line in re.split(r"\r?\n", text):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif ": " in line:
            k, _, v = line.partition(": ")
            header[f"{section} › {k}" if section else k] = v
            header.setdefault(k, v)
    rows, cols = int(header["Number of rows"]), int(header["Number of columns"])
    kind = header.get("Image Data Type", "short" if path.lower().endswith(".top") else "double").lower()
    dtype = {"double": "<f8", "float": "<f4", "short": "<i2", "integer": "<i4", "int": "<i4"}.get(kind, "<f8")
    data = np.fromfile(path, dtype=dtype, count=rows * cols, offset=size).reshape(rows, cols).astype(float)

    def amplitude(key):
        v = header.get(key, "")
        nums = _floats(v)
        return (nums[0], v.split()[-1]) if nums and v.split() else (None, "")

    xa, xu = amplitude("X Amplitude")
    ya, yu = amplitude("Y Amplitude")
    za, zu = amplitude("Z Amplitude")
    unit = zu or "nm"
    if dtype == "<i2" and za is not None and data.max() > data.min():
        # integer data spans the Z amplitude (same convention as AFMReader)
        data = (data - data.min()) / (data.max() - data.min()) * za
    channel = Channel(header.get("Acquisition channel", "Topography"), data, unit,
                      length_to_nm(xa / cols, xu) if xa else None, length_to_nm(ya / rows, yu) if ya else None)
    md = OrderedDict()
    for key in ("Acquisition time", "Head type", "Acquisition channel", "X Amplitude", "Y Amplitude",
                "Z Amplitude", "Set Point", "Topography Bias", "X-Frequency", "Angle", "Force constant",
                "Resonance frequency", "Y scanning direction"):
        if header.get(key):
            md[key] = header[key]
    return MicroscopyFile(path, [channel], md, header)


# =============================================================================
# Gatan / FEI / TIFF via RosettaSciIO
# =============================================================================

def _flatten(d, prefix=""):
    out = OrderedDict()
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif not isinstance(v, (bytes, np.ndarray)) or (isinstance(v, np.ndarray) and v.size <= 16):
            out[key] = str(v)
    return out


_INTERESTING_TAGS = ("Voltage", "Indicated Magnification", "Actual Magnification", "Operation Mode",
                     "Microscope Info.Name", "Acquisition Date", "Acquisition Time", "Exposure (s)",
                     "Exposure Time (s)", "Camera Length", "Spot size", "STEM Camera Length",
                     "beam_energy", "magnification", "camera_length", "dwell_time", "date", "time", "title",
                     "HV", "WD", "Mag", "Detector", "PixelWidth", "Pixel Size", "EHT", "Working Distance")


@register_format("Gatan / FEI / TIFF", (".dm3", ".dm4", ".emd", ".ser", ".tif", ".tiff"))
def _load_rsciio(path):
    import importlib

    ext = os.path.splitext(path)[1].lower()
    module = {".dm3": "digitalmicrograph", ".dm4": "digitalmicrograph", ".emd": "emd", ".ser": "tia",
              ".tif": "tiff", ".tiff": "tiff"}[ext]
    try:
        signals = importlib.import_module(f"rsciio.{module}").file_reader(path)
    except Exception:
        if module != "tiff":
            raise
        return _load_plain_image(path)

    channels, md, header, notes = [], OrderedDict(), OrderedDict(), []
    for k, sig in enumerate(signals):
        data = np.asarray(sig["data"])
        if data.dtype.names:                                     # RGB(A) structured array
            data = np.mean([data[n] for n in data.dtype.names[:3]], axis=0)
        axes = sig["axes"]
        if data.ndim < 2 or len(axes) < 2:
            continue
        ay, ax = axes[-2], axes[-1]
        units = (str(ax.get("units", "")), str(ay.get("units", "")))
        px_x, px_y = length_to_nm(ax.get("scale"), units[0]), length_to_nm(ay.get("scale"), units[1])
        if module == "tiff" and units[0].lower() in ("in", "inch", "inches", "cm", "mm"):
            px_x = px_y = None                                   # print resolution, not a calibration
        if px_x is None and units[0] and units[0] not in ("<undefined>", "px", "pixel"):
            notes.append(f"Calibration {ax.get('scale'):.5g} {units[0]}/px is not a real-space length")
        title = sig.get("metadata", {}).get("General", {}).get("title") or f"Signal {k}"
        frames = data.reshape(-1, data.shape[-2], data.shape[-1])
        for f_i, frame in enumerate(frames[:64]):
            name = title if len(frames) == 1 else f"{title} [{f_i}]"
            if any(c.name == name for c in channels):
                name = f"{name} ({k})"
            channels.append(Channel(name, frame.astype(float), sig.get("metadata", {}).get("Signal", {})
                                    .get("quantity", ""), px_x, px_y))
        flat = _flatten(sig.get("metadata", {}))
        flat.update(_flatten(sig.get("original_metadata", {})))
        for key, value in flat.items():
            header.setdefault(key, value)
            if any(key.endswith(t) or key.split(".")[-1] == t for t in _INTERESTING_TAGS):
                md.setdefault(key.split(".")[-1] if key.split(".")[-1] not in md else key, value)
    for n in notes[:1]:
        md["Note"] = n
    return MicroscopyFile(path, channels, md, header)


# =============================================================================
# Plain images
# =============================================================================

def _scale_from_metadata_csv(path):
    """nm/px for an image exported by ScanBound (metadata.csv next to it), else None."""
    import csv
    meta = os.path.join(os.path.dirname(path), "metadata.csv")
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("Filename") == os.path.basename(path):
                    x = float(row.get("Pixel Scale X (nm)") or "nan")
                    y = float(row.get("Pixel Scale Y (nm)") or x)
                    if math.isfinite(x) and x > 0:
                        return x, (y if math.isfinite(y) and y > 0 else x)
    except (OSError, ValueError):
        return None
    return None


@register_format("Image", (".png", ".jpg", ".jpeg", ".bmp"))
def _load_plain_image(path):
    from PIL import Image

    with Image.open(path) as img:
        mode = img.mode
        data = np.asarray(img.convert("F"), dtype=float)
        size = img.size
    scale = _scale_from_metadata_csv(path)
    px_x, px_y = scale if scale else (None, None)
    md = OrderedDict([("Image mode", mode), ("Size", f"{size[0]} × {size[1]}"),
                      ("Scale source", "metadata.csv" if scale else "none (set manually)")])
    return MicroscopyFile(path, [Channel("Intensity", data, "", px_x, px_y)], md, OrderedDict(md))
