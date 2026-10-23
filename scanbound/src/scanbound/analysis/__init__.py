"""
Statistics and charts over many annotations (composition, coverage, size, orientation ...).

    roi       the analyses themselves (registry; Qt-free; unchanged from the old app)
    gnr       nanoribbons: split into ribbons, length / width, branching and fusion, herringbone alignment
    adapter   builds their input from a Project: ground truth only, physical scale, reference axes

    from scanbound.analysis import analysis_context, run_analyses
    ctx = analysis_context(project, image_ids)
    results = run_analyses(ctx, ["composition", "size"])
"""

from . import gnr as _gnr  # noqa: F401  (registers the nanoribbon analyses)
from .adapter import analysis_context, analysis_data
from .roi import ANALYSES, AnalysisContext, AnalysisData, AnalysisResult, ImageInfo, Param, default_params, register_analysis, run_analyses

__all__ = ["analysis_context", "analysis_data", "ANALYSES", "AnalysisContext", "AnalysisData", "AnalysisResult",
           "ImageInfo", "Param", "default_params", "register_analysis", "run_analyses"]
