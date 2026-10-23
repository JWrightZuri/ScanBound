"""File loading: every supported microscopy format into one :class:`Channel` structure."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .microscopy import (Channel, MicroscopyFile, find_scans, fmt_length, fmt_si, format_for, length_to_nm,
                         load_file, register_format, supported_extensions)

from .data import (ChannelInfo, DataFile, DataFormatError, is_data_file, list_data_files, load_stack,  # noqa: E402
                   open_data)
from .nanonis_dat import load_nanonis_dat  # noqa: E402

__all__ = ["ChannelInfo", "DataFile", "DataFormatError", "is_data_file", "list_data_files", "load_stack",
           "open_data", "load_nanonis_dat",
           "Channel", "MicroscopyFile", "LoadedImage", "find_scans", "fmt_length", "fmt_si", "format_for",
           "length_to_nm", "load_file", "load_image", "register_format", "supported_extensions"]


@dataclass
class LoadedImage:
    path: str
    channel: str
    data: np.ndarray                # 2D float, row 0 = top
    unit: str
    px_scale: Optional[float]       # nm per pixel (mean of x and y), None = uncalibrated

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


def load_image(path: Union[str, "os.PathLike[str]"], channel: Optional[str] = None,
               px_scale: Optional[float] = None) -> LoadedImage:
    """
    One channel of a file (the topography / default channel unless ``channel`` is given).
    ``px_scale`` (nm per pixel) overrides the file's calibration.
    """
    f = load_file(os.fspath(path))
    if not f.channels:
        raise ValueError(f"{path}: the file contains no images")
    ch = f.channel(channel) if channel else f.default()
    if ch is None:
        raise KeyError(f"{path}: no channel {channel!r}; available: {', '.join(f.channel_names())}")
    scale = px_scale
    if scale is None and ch.calibrated:
        scale = (float(ch.px_size_x) + float(ch.px_size_y)) / 2.0
    return LoadedImage(f.path, ch.name, ch.data, ch.unit, scale)
