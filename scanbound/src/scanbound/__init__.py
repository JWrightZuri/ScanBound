"""
ScanBound detection core
========================

UI-independent pipeline for finding nanostructures (nanoribbons, particles, islands ...) in
SPM / microscopy images::

    from scanbound import Detector, DetectionConfig

    det = Detector(DetectionConfig.load("image_adapt_params.json"))
    result = det.detect_file("scan_0001.sxm")
    result.features            # DataFrame: area, length, width, orientation ... per object
    result.labeled_objects()   # [(SegObject with ROI geometry, label)]

Layers (each only imports the ones above it):

    steps/        processing steps + registries (adapt: gray -> mask, spm: physical units)
    pipeline      ordered steps, caching, error reporting
    roi           mask -> one ROI (rect / polygon / centre line / point) per object
    measure       shape features per object
    labeling      features -> labels (single / rules / bins / k-means clusters)
    ribbons       skeleton segments: ribbon lengths and widths
    config        typed, validated settings; reads / writes the dialog's JSON files
    detect        Detector: all of the above for an image, a file or a batch
    io, export    file formats in; CSV / JSON / overlay PNG out
    cli           ``scanbound detect ...``
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import DetectionConfig, PipelineConfig, RibbonSettings, StepConfig
from .detect import BatchItem, Detection, Detector, NoMaskError
from .errors import ConfigError, ParamError, PipelineError, ScanBoundError, UnknownStepError
from .labeling import LabelingConfig, assign_labels
from .measure import FEATURES, measure_objects
from .pipeline import Pipeline, PipelineCache, PipelineResult, run_pipeline
from .ribbons import Ribbon, find_ribbons
from .roi import ROI_TYPES, RoiSettings, SegObject, extract_objects
from .steps import ADAPT_STEPS, SPM_STEPS, Param, StepRegistry, StepSpec

__all__ = [
    "__version__",
    "DetectionConfig", "PipelineConfig", "RibbonSettings", "StepConfig",
    "BatchItem", "Detection", "Detector", "NoMaskError",
    "ConfigError", "ParamError", "PipelineError", "ScanBoundError", "UnknownStepError",
    "LabelingConfig", "assign_labels", "FEATURES", "measure_objects",
    "Pipeline", "PipelineCache", "PipelineResult", "run_pipeline",
    "Ribbon", "find_ribbons", "ROI_TYPES", "RoiSettings", "SegObject", "extract_objects",
    "ADAPT_STEPS", "SPM_STEPS", "Param", "StepRegistry", "StepSpec",
]
