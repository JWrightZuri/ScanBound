"""
Machine learning on annotations: dataset export, training, prediction as proposals, evaluation.

    render       the picture a model sees (shared by export and prediction, so they always match)
    dataset      AnnotationStore ground truth -> YOLO dataset with a manifest (traceable to raw data)
    train        Ultralytics training wrapper that records how its data was made
    predict      model -> *proposed* annotations with confidences
    evaluate     IoU matching, precision / recall / F1 / AP per class, review statistics

Only ``train`` and ``predict`` need ultralytics; it is imported when they are used.
"""

from .dataset import DatasetReport, export_yolo
from .evaluate import ClassMetrics, Evaluation, evaluate, geometry_iou, match, review_stats
from .render import RENDER_MODES, RenderSpec, render_for_model

__all__ = ["DatasetReport", "export_yolo", "ClassMetrics", "Evaluation", "evaluate", "geometry_iou", "match",
           "review_stats", "RENDER_MODES", "RenderSpec", "render_for_model"]
