"""
Scientific data model, independent of any instrument and of the GUI.

    Axis, Calibration          what each array dimension means, in which unit, and how sure we are
    Image, ImageStack          calibrated 2D data (NaN = no data), immutable, with history
    Spectrum, Signal           point spectroscopy on a sweep axis
    SourceRef, ProcessingRecord  provenance: raw file + every operation since
    Quantity                   value + unit + uncertainty
"""

from .axes import CALIBRATION_SOURCES, DEFAULT_SCANNER_REL_UNCERTAINTY, UNCALIBRATED, Axis, Calibration
from .image import Image, ImageStack, ImageStats, QualityIssue, ScanFrame
from .provenance import (EFFECTS, ProcessingRecord, SourceRef, array_digest, file_digest, history_changes_geometry,
                         params_digest)
from .spectrum import Signal, Spectrum, average, split_column
from .units import Quantity, convert, format_value, split_unit, to_base

__all__ = [
    "CALIBRATION_SOURCES", "DEFAULT_SCANNER_REL_UNCERTAINTY", "UNCALIBRATED", "Axis", "Calibration",
    "Image", "ImageStack", "ImageStats", "QualityIssue", "ScanFrame",
    "EFFECTS", "ProcessingRecord", "SourceRef", "array_digest", "file_digest", "history_changes_geometry",
    "params_digest", "Signal", "Spectrum", "average", "split_column",
    "Quantity", "convert", "format_value", "split_unit", "to_base",
]
