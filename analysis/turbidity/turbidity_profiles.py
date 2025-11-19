import sys
from pathlib import Path

import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TURBIDITY_PARAMS
from analysis.turbidity.preprocessing import preprocess_img


@dataclass
class TurbidityProfile:
    """Container for turbidity analysis results."""
    raw_profile: np.ndarray
    normalized_profile: np.ndarray
    excluded_regions: Dict[str, Any]
    gradient: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    centerline_x: Optional[int] = None

    @property
    def full_image_height(self) -> int:
        return self.excluded_regions['analysis_height']

def normalize_to_full_image(y_abs: int, profile: TurbidityProfile) -> float:
    """Convert absolute pixel to normalized 0–1 over the full image"""
    full_h = profile.excluded_regions['analysis_height']
    return y_abs / full_h

# turbidity computation
def _compute_turbidity_profile(
        gray: np.ndarray,
        prep,
        apply_horizontal_exclusion: bool,
        extra_excluded_info: Optional[Dict[str, Any]] = None,
) -> TurbidityProfile:
    """
    Compute turbidity profile with region exclusion (core for both full width and centerline)
    """
    # multiple channel analysis
    saturation = prep.hsv[:, :, 1]
    lightness = prep.lab[:, :, 0]

    # exclusion regions
    # top exclusion
    top_exclude_idx = int(prep.height * TURBIDITY_PARAMS['top_exclude_fraction'])

    # bottom exclusion
    bottom_exclude_idx = int(prep.height * (1 - TURBIDITY_PARAMS['bottom_exclude_fraction']))

    # horizontal exclusions
    if apply_horizontal_exclusion:
        left_exclude_idx = int(prep.width * TURBIDITY_PARAMS['left_exclude_fraction'])
        right_exclude_idx = int(prep.width * (1 - TURBIDITY_PARAMS['right_exclude_fraction']))
        gray_use = gray[:, left_exclude_idx:right_exclude_idx]
        saturation_use = saturation[:, left_exclude_idx:right_exclude_idx]
        lightness_use = lightness[:, left_exclude_idx:right_exclude_idx]
    else:
        left_exclude_idx = 0
        right_exclude_idx = prep.width
        gray_use = gray
        saturation_use = saturation
        lightness_use = lightness

    # recompute profiles after exclusions
    excluded_profiles = {
        'intensity': np.mean(gray_use, axis=1),
        'saturation': np.mean(saturation_use, axis=1),
        'lightness': np.mean(lightness_use, axis=1),
        'variance': np.var(gray_use, axis=1)
    }

    # update profile with excluded intensity
    raw_profile = excluded_profiles['intensity']

    # analysis region
    analysis_region = raw_profile[top_exclude_idx:bottom_exclude_idx]

    # normalize profile
    if len(analysis_region) > 0:
        min_val = analysis_region.min()
        max_val = analysis_region.max()
        if max_val > min_val:
            normalized = (analysis_region - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(analysis_region)
    else:
        normalized = np.array([])

    # normalized profile with excluded regions
    full_normalized = np.zeros_like(raw_profile)
    if len(normalized) > 0:
        full_normalized[top_exclude_idx:bottom_exclude_idx] = normalized

    # base excluded info
    excluded_info = {
        'top_exclude_idx': top_exclude_idx,
        'bottom_exclude_idx': bottom_exclude_idx,
        'analysis_height': prep.height,
        'excluded_top_fraction': top_exclude_idx / prep.height,
        'excluded_bottom_fraction': (prep.height - bottom_exclude_idx) / prep.height,
        'left_exclude_idx': left_exclude_idx,
        'right_exclude_idx': right_exclude_idx,
        'excluded_left_fraction': left_exclude_idx / prep.width,
        'excluded_right_fraction': (prep.width - right_exclude_idx) / prep.width,
    }

    # merge extra metadata
    if extra_excluded_info:
        excluded_info.update(extra_excluded_info)

    return TurbidityProfile(
        raw_profile=raw_profile,
        normalized_profile=full_normalized,
        excluded_regions=excluded_info
    )


# full-width turbidity profile
def compute_turbidity_profile(image: np.ndarray) -> TurbidityProfile:
    """
    Compute turbidity profile for the full vial region with left/right exclusion
    """
    # preprocess img
    prep = preprocess_img(image)

    # compute turbidity profile
    return _compute_turbidity_profile(
        gray=prep.gray,
        prep=prep,
        apply_horizontal_exclusion=True,
        extra_excluded_info=None
    )


# centerline turbidity profile
def compute_centerline_turbidity_profile(
        image: np.ndarray,
        centerline_width: int = TURBIDITY_PARAMS['centerline_width'],
) -> TurbidityProfile:
    """
    Compute turbidity profile along the centerline strip of the vial
    """
    # preprocess img
    prep = preprocess_img(image)

    # centerline calculation
    center_x = int(prep.width / 2)
    half_width = centerline_width // 2
    center_left_bound = max(0, center_x - half_width)
    center_right_bound = min(prep.width, center_x + half_width)

    # centerline region
    center_gray = prep.gray[:, center_left_bound:center_right_bound]

    # store metadata
    extra_info = {
        'centerline_x': center_x,
        'centerline_width': centerline_width,
        'centerline_bounds': (center_left_bound, center_right_bound)
    }

    return _compute_turbidity_profile(
        gray=center_gray,
        prep=prep,
        apply_horizontal_exclusion=False,
        extra_excluded_info=extra_info
    )


def compute_variance_between_changes(
        profile: TurbidityProfile,
        change_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute brightness variance in the analysis region and in segments
    defined by successive sudden brightness changes.

    The segments are:
        [start_of_analysis → first_change_start),
        [first_change_start → second_change_start), ...,
        [last_change_start → end_of_analysis)

    All variances are computed on the normalized profile.
    """
    norm = profile.normalized_profile
    ex = profile.excluded_regions

    top_idx = ex['top_exclude_idx']
    bottom_idx = ex['bottom_exclude_idx']
    height = ex['analysis_height']

    if bottom_idx <= top_idx:
        return {"overall_variance": 0.0, "segments": []}

    # exclude top and bottom
    analysis_region = norm[top_idx:bottom_idx]
    H = len(norm)  # full centerline length

    # absolute height normalized 0–1 across image
    z_full = np.linspace(top_idx / H, bottom_idx / H, len(analysis_region))

    if len(analysis_region) > 1:
        overall_var = float(np.var(analysis_region))
    else:
        overall_var = 0.0

    # Build segment boundaries from sudden-change start positions
    if not change_events:
        boundaries = [top_idx, bottom_idx]
    else:
        starts = []
        for ev in change_events:
            sa = int(ev.get("start_absolute", -1))
            # keep only those inside the analysis region
            if sa <= top_idx or sa >= bottom_idx:
                continue
            starts.append(sa)

        if not starts:
            boundaries = [top_idx, bottom_idx]
        else:
            starts = sorted(set(starts))
            boundaries = [top_idx] + starts + [bottom_idx]

    segments = []
    for seg_id, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if e <= s:
            continue
        seg_data = norm[s:e]
        if len(seg_data) < 2:
            continue

        var = float(np.var(seg_data))
        mean = float(np.mean(seg_data))
        std = float(np.std(seg_data))

        segments.append({
            "segment_id": seg_id,
            "start_absolute": int(s),
            "end_absolute": int(e),
            "start_normalized": float(s / height),
            "end_normalized": float(e / height),
            "height_normalized": float(e - s) / height,
            "z_full": z_full[s:e].tolist(),
            "height_pixels": int(e - s),
            "mean_brightness": mean,
            "variance": var,
            "std_brightness": std,
        })

    return {
        "overall_variance": overall_var,
        "segments": segments,
    }
