# analysis/gelled_analysis.py
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any

from config import CURVE_PARAMS
from image_analysis.guided_curve_analysis import CurveAnalyzer
from image_analysis.guided_curve_tracer import GuidedCurveTracer

def run_curve_metrics(crop_path: Path) -> Dict[str, Any]:
    """
    Analyzes a cropped image to determine if it shows a gelled state based on curve characteristics.

    Returns:
      {
        "gelled_by_curve": bool,
        "stats": {... CurveStatistics ...},
        "curve_metadata": {...},
        "reason": str
      }
    """
    img = cv2.imread(str(crop_path))
    if img is None:
        return {"gelled_by_curve": False, "stats": {}, "curve_metadata": {}, "reason": "image_load_failed"}

    # H, W = img.shape[:2]
    params = CURVE_PARAMS

    tracer = GuidedCurveTracer(
        vertical_bounds=params.get("vertical_bounds", (0.30, 0.80)),
        horizontal_bounds=params.get("horizontal_bounds", (0.05, 0.95)),
        search_offset_frac=params.get("search_offset_frac", 0.05),
        median_kernel=params.get("median_kernel", 9),
        max_step_px=params.get("max_step_px", 4),
    )

    xs, ys, meta = tracer.trace_curve(img, crop_path, guide_y=None)
    if len(xs) < params.get("min_points", 50):
        return {
            "gelled_by_curve": False,
            "stats": {},
            "curve_metadata": meta,
            "reason": f"insufficient_points:{len(xs)}"
        }

    xs_np = np.asarray(xs, np.float32)
    ys_np = np.asarray(ys, np.float32)

    analyzer = CurveAnalyzer()
    stats = analyzer.compute_comprehensive_statistics(
        xs=xs_np, ys=ys_np, baseline_y=None, window_size=20
    )
    # initialise segments data
    segments = analyzer.segment_curve(
        xs=xs_np,
        ys=ys_np,
        num_segments=params.get("num_segments", 5),
    )
    # decision: variance from baseline
    var_thr = params.get("gel_variance_thr", 80.0)
    std_dev_thr = params.get("std_dev_thr", 10.0)
    inter_segment_variance_thr = params.get("inter_segment_variance", 40.0)
    rough_thr = params.get("roughness_thr", 0.85)

    gelled = (stats.variance_from_baseline >= var_thr)

    return {
        "gelled_by_curve": bool(gelled),
        "stats": {k: getattr(stats, k) for k in stats.__dataclass_fields__.keys()},
        "curve_metadata": meta,
        "reason": f"variance {stats.variance_from_baseline:.2f} > {var_thr} threshold, "
                  f"std_dev {stats.std_dev_from_baseline:.2f} > {std_dev_thr} threshold, "
                  f"inter_segment_variance {segments['inter_segment_variance']:.2f} > {inter_segment_variance_thr} threshold, "
                  f"roughness {stats.roughness:.2f} > {rough_thr} threshold" if gelled else "no curve"
    }
