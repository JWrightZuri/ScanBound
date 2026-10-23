import numpy as np

from scanbound.roi import ROI_PARAMS, RoiSettings, extract_objects


def test_zero_padding_is_allowed_and_encloses():
    assert next(p for p in ROI_PARAMS if p.key == "padding").minimum == 0
    m = np.zeros((40, 40), bool)
    m[10:30, 10:20] = 1
    for tol in (0.0, 2.0):
        o = extract_objects(m, RoiSettings(padding=0, tolerance=tol, max_vertices=0))[0]
        x0, y0 = o.points.min(0)
        x1, y1 = o.points.max(0)
        assert x0 <= 10 and y0 <= 10 and x1 >= 20 and y1 >= 30
        assert x0 > 10 - tol - 2 and x1 < 20 + tol + 2                   # tighter than the old 0.5 px minimum


def test_roi_analysis_can_leave_out_objects_touching_the_border():
    import pandas as pd
    from scanbound.analysis.roi import AnalysisContext, AnalysisData, ImageInfo
    rows = [{"objectName": n, "type": "RectROI", "x1": x, "y1": 10, "dx": 10, "dy": 10, "file": "a", "label": "m"}
            for n, x in (("inner", 20), ("edge", 0), ("right", 90))]
    data = AnalysisData(pd.DataFrame(rows), {"a": ImageInfo("a", 100, 50)})
    assert len(AnalysisContext(data).objects) == 3
    ctx = AnalysisContext(data, exclude_border=True)
    assert list(ctx.objects["objectName"]) == ["inner"] and len(ctx.shapes) == 1 and ctx.notes
