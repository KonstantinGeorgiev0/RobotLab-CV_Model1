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

    params = CURVE_PARAMS

    tracer = GuidedCurveTracer(
        vertical_bounds=params.get("vertical_bounds", (0.30, 0.80)),
        horizontal_bounds=params.get("horizontal_bounds", (0.05, 0.95)),
        search_offset_frac=params.get("search_offset_frac", 0.10),
        median_kernel=params.get("median_kernel", 9),
        max_step_px=params.get("max_step_px", 4),
    )

    xs, ys, meta = tracer.trace_curve(img, crop_path, guide_y=None)
    min_pts = params.get("min_points", 50)
    if len(xs) < min_pts:
        return {
            "gelled_by_curve": False,
            "stats": {},
            "curve_metadata": meta,
            "reason": f"insufficient_points:{len(xs)}<{min_pts}"
        }

    xs_np = np.asarray(xs, np.float32)
    ys_np = np.asarray(ys, np.float32)

    analyzer = CurveAnalyzer()
    stats = analyzer.compute_comprehensive_statistics(
        xs=xs_np, ys=ys_np, baseline_y=None, window_size=20
    )

    # Thresholds
    yvar_thr = params.get("y_variance_thr", 100.0)                    # strong separator
    std_dev_thr = params.get("std_dev_thr", 10.0)                     # secondary
    inter_seg_thr = params.get("inter_segment_variance_thr", 80.0)    # spatial stability
    loc_var_mean_thr = params.get("local_variance_mean_thr", 30.0)    # micro-waviness
    spec_energy_thr = params.get("spectral_energy_thr", 5_000_000.0)  # frequency content
    votes_needed = params.get("votes_needed", 2)                      # majority of rules

    # Collect metrics safely
    y_variance = getattr(stats, "variance", np.nan)
    std_from_base = getattr(stats, "std_dev_from_baseline", np.nan)
    inter_segment_variance = getattr(stats, "inter_segment_variance", np.nan)
    local_variance_mean = getattr(stats, "local_variance_mean", np.nan)
    spectral_energy = getattr(stats, "spectral_energy", np.nan)

    # Rules & votes
    rule_hits = []

    if not np.isnan(y_variance) and y_variance >= yvar_thr:
        rule_hits.append(f"y_variance {y_variance:.2f} ≥ {yvar_thr}")
    if not np.isnan(std_from_base) and std_from_base >= std_dev_thr:
        rule_hits.append(f"std_dev {std_from_base:.2f} ≥ {std_dev_thr}")
    if not np.isnan(inter_segment_variance) and inter_segment_variance >= inter_seg_thr:
        rule_hits.append(f"inter_segment_variance {inter_segment_variance:.2f} ≥ {inter_seg_thr}")
    if not np.isnan(local_variance_mean) and local_variance_mean >= loc_var_mean_thr:
        rule_hits.append(f"local_variance_mean {local_variance_mean:.2f} ≥ {loc_var_mean_thr}")
    if not np.isnan(spectral_energy) and spectral_energy >= spec_energy_thr:
        rule_hits.append(f"spectral_energy {spectral_energy:.2e} ≥ {spec_energy_thr:.2e}")

    gelled = (len(rule_hits) >= votes_needed)

    reason = (
        "gel" if gelled else "no_gel"
    ) + " | hits: " + (", ".join(rule_hits) if rule_hits else "none")

    return {
        "gelled_by_curve": bool(gelled),
        "stats": {k: getattr(stats, k) for k in stats.__dataclass_fields__.keys()},
        "curve_metadata": meta,
        "reason": reason
    }
