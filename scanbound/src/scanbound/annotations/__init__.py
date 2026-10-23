"""
Annotations: one geometry framework, many meanings, independent of the GUI.

    geometry     Point, Segment, Polyline, Polygon, Rect, Ellipse (pixel coordinates)
    kinds        feature, centre_line, measurement_line, orientation_vector, reference_axis,
                 lattice_vector ... with validation rules (register_kind to add more)
    annotation   Annotation (geometry + kind + label + review status + origin)
    store        AnnotationStore: undoable edits, change events, queries, hit testing
    labels       LabelSet: label names, colours, number keys
    legacy       the old annotations.csv format
"""

from .annotation import GROUND_TRUTH, STATUSES, Annotation, Origin, new_id
from .geometry import (GEOMETRY_TYPES, Ellipse, Geometry, Mask, Point, Polygon, Polyline, Rect, Segment, angle_deg, bbox_iou,
                       clip_to_image, geometry_from_dict)
from .kinds import KINDS, AnnotationKind, Issue, ValidationContext, kind, register_kind, validate
from .labels import LabelSet
from .store import AnnotationStore, Change, StoreEvent

__all__ = [
    "GROUND_TRUTH", "STATUSES", "Annotation", "Origin", "new_id",
    "GEOMETRY_TYPES", "Ellipse", "Geometry", "Mask", "Point", "Polygon", "Polyline", "Rect", "Segment", "angle_deg",
    "bbox_iou", "clip_to_image", "geometry_from_dict",
    "KINDS", "AnnotationKind", "Issue", "ValidationContext", "kind", "register_kind", "validate",
    "LabelSet", "AnnotationStore", "Change", "StoreEvent",
]
