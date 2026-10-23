"""
Nanonis spectroscopy files (``.dat``): bias spectroscopy, Z spectroscopy, history data ...

Layout: ``key<TAB>value<TAB>`` header lines, a ``[DATA]`` line, one tab-separated line of column
titles (``"Bias calc (V)"``, ``"Current (A) [bwd]"`` ...) and numeric rows.  The first column is
the sweep axis.  Missing header entries are tolerated; a file without ``[DATA]`` or without any
numeric row is rejected with a readable message.
"""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..errors import ScanBoundError
from ..model import Axis, Calibration, Signal, SourceRef, Spectrum, split_column


class SpectrumFormatError(ScanBoundError, ValueError):
    """A spectroscopy file could not be read."""


def _float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        v = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def is_nanonis_dat(head: bytes) -> bool:
    return head.startswith(b"Experiment\t") or b"\n[DATA]" in head[:65536] and b"\t" in head[:200]


def read_header(path: str) -> tuple["OrderedDict[str, str]", int]:
    """(header, byte offset of the column title line)."""
    header: "OrderedDict[str, str]" = OrderedDict()
    with open(path, "rb") as fh:
        raw = fh.read()
    pos = raw.find(b"[DATA]")
    if pos < 0:
        raise SpectrumFormatError(f"{os.path.basename(path)}: no [DATA] section (not a Nanonis spectroscopy file?)")
    for line in raw[:pos].decode("latin-1").splitlines():
        if "\t" in line:
            key, _, value = line.partition("\t")
            header[key.strip()] = value.rstrip("\t").strip()
    nl = raw.find(b"\n", pos)
    return header, (nl + 1 if nl >= 0 else len(raw))


def load_nanonis_dat(path: str) -> Spectrum:
    header, start = read_header(path)
    with open(path, "rb") as fh:
        fh.seek(start)
        text = fh.read().decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SpectrumFormatError(f"{os.path.basename(path)}: the [DATA] section is empty")
    titles = [t.strip() for t in lines[0].rstrip("\t").split("\t")]
    try:
        table = np.genfromtxt(io.StringIO("\n".join(lines[1:])), delimiter="\t", dtype=float,
                              invalid_raise=False, missing_values=("", "NaN", "N/A"), filling_values=np.nan)
    except ValueError as exc:
        raise SpectrumFormatError(f"{os.path.basename(path)}: unreadable data ({exc})") from exc
    table = np.atleast_2d(table)
    if table.size == 0 or table.shape[0] == 0:
        raise SpectrumFormatError(f"{os.path.basename(path)}: no numeric data rows")
    if table.shape[1] < len(titles):        # trailing tabs / short rows
        table = np.hstack([table, np.full((table.shape[0], len(titles) - table.shape[1]), np.nan)])
    table = table[:, : len(titles)]
    table = table[np.isfinite(table[:, 0])]  # rows without a sweep value carry nothing
    if table.shape[0] == 0:
        raise SpectrumFormatError(f"{os.path.basename(path)}: the sweep column has no values")

    name, unit, _ = split_column(titles[0])
    axis = Axis.from_values(name.split()[0].lower() if name else "x", table[:, 0], unit, label=name,
                            calibration=Calibration("file", "sweep values written by the instrument"))
    signals = []
    for j, title in enumerate(titles[1:], start=1):
        n, u, tag = split_column(title)
        signals.append(Signal(n, u, table[:, j].copy(), tag))

    pos = tuple(_float(header.get(k)) for k in ("X (m)", "Y (m)", "Z (m)"))
    position = pos if pos[0] is not None and pos[1] is not None else None   # type: ignore[assignment]
    md = OrderedDict()
    for key in ("Experiment", "Saved Date", "Start time", "Bias>Bias (V)", "Z-Ctrl hold", "Integration time (s)",
                "Settling time (s)", "Lock-in>Lock-in status", "Lock-in>Amplitude", "Lock-in>Frequency (Hz)",
                "Bias Spectroscopy>Number of sweeps", "Bias Spectroscopy>backward sweep", "Comment"):
        if header.get(key):
            md[key] = header[key]
    return Spectrum(axis, signals, position, header.get("Experiment", ""),
                    SourceRef.for_file(path, "Nanonis spectroscopy"), {"summary": md, "header": dict(header)})
