import sys
from pathlib import Path

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import TURBIDITY_PARAMS
from analysis.turbidity.turbidity_profiles import TurbidityProfile, normalize_to_full_image


def segment_brightness_regions(
        profile: TurbidityProfile,
        similarity_threshold: float = TURBIDITY_PARAMS["similarity_threshold"],
        min_region_size: int = TURBIDITY_PARAMS["min_region_size"],
        smoothing_sigma: float = TURBIDITY_PARAMS["smoothing_sigma"],
        gradient_threshold: float = TURBIDITY_PARAMS["gradient_threshold"],
) -> List[Dict[str, Any]]:
    """
    Segment the turbidity profile into regions of similar brightness.

    Args:
        profile: TurbidityProfile object
        similarity_threshold: max std dev within a region to be "homogeneous"
        min_region_size: minimum pixels per region
        smoothing_sigma: Gaussian sigma
        gradient_threshold: used as brightness-change threshold
    """
    norm_profile = profile.normalized_profile
    excluded = profile.excluded_regions

    # analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    print("\nSTART IDX: ", start_idx, "\nEND IDX: ", end_idx, "\n")
    excluded_analysis_region = norm_profile[start_idx:end_idx]

    if len(excluded_analysis_region) < min_region_size:
        return []

    # smooth profile to reduce noise
    smoothed = gaussian_filter1d(excluded_analysis_region, sigma=smoothing_sigma)

    gradient = np.gradient(smoothed)
    abs_gradient = np.abs(gradient)

    # brightness-change based boundaries
    # full_height = excluded['analysis_height']
    # y_positions = np.linspace(0, 1, full_height)

    change_threshold = TURBIDITY_PARAMS.get(
        "brightness_change_threshold",
        gradient_threshold,
    )
    window = max(3, min_region_size // 2)
    change_points = _find_brightness_change_points(
        smoothed=smoothed,
        change_threshold=change_threshold,
        min_distance=min_region_size,
        window=window,
    )

    # build segments between change points
    boundaries = [0] + change_points.tolist() + [len(excluded_analysis_region)]

    regions: List[Dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        region_height = seg_end - seg_start
        if region_height < min_region_size:
            continue

        segment_data = smoothed[seg_start:seg_end]
        raw_segment_data = excluded_analysis_region[seg_start:seg_end]

        mean_brightness = float(np.mean(segment_data))
        std_brightness = float(np.std(segment_data))
        variance = float(np.var(segment_data))
        min_brightness = float(np.min(segment_data))
        max_brightness = float(np.max(segment_data))

        # gradient stats
        segment_gradient = abs_gradient[seg_start:seg_end]
        mean_gradient = float(np.mean(segment_gradient))
        max_gradient = float(np.max(segment_gradient))

        # normalized positions in excluded image
        start_normalized_after_exclusion_region = seg_start / len(excluded_analysis_region)
        end_normalized_after_exclusion_region = seg_end / len(excluded_analysis_region)
        center_normalized_excluded_region = (start_normalized_after_exclusion_region + end_normalized_after_exclusion_region) / 2.0

        # absolute positions in excluded image
        start_absolute_after_exclusion_region = start_idx + seg_start
        end_absolute_after_exclusion_region = start_idx + seg_end
        center_absolute_after_exclusion_region = (start_absolute_after_exclusion_region + end_absolute_after_exclusion_region) / 2.0

        # norm pos in original image
        start_normalized_original_image = (start_idx + seg_start) / len(norm_profile)
        end_normalized_original_image = (start_idx + seg_end) / len(norm_profile)

        center_normalized_original_image = (start_normalized_original_image + end_normalized_original_image) / 2.0

        # abs pos in original image
        start_absolute = start_idx + seg_start
        end_absolute = start_idx + seg_end
        center_absolute = (start_absolute + end_absolute) / 2.0

        regions.append({
            'region_id': i,
            'start_normalized_analysis_region': float(start_normalized_after_exclusion_region),
            'end_normalized_analysis_region': float(end_normalized_after_exclusion_region),
            'center_normalized_analysis_region': float(center_normalized_excluded_region),
            'start_absolute_analysis_region': int(start_absolute_after_exclusion_region),
            'end_absolute_analysis_region': int(end_absolute_after_exclusion_region),
            'center_absolute_analysis_region': int(center_absolute_after_exclusion_region),
            'start_normalized': float(start_normalized_original_image),
            'end_normalized': float(end_normalized_original_image),
            'center_normalized': float(center_normalized_original_image),
            'start_absolute': int(start_absolute),
            'end_absolute': int(end_absolute),
            'height_pixels': int(region_height),
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'variance': variance,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'peak_brightness': max_brightness,
            'brightness_range': float(max_brightness - min_brightness),
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'is_homogeneous': std_brightness < similarity_threshold,
            'raw_values': raw_segment_data.tolist(),
            'smoothed_values': segment_data.tolist(),
        })

    return regions


def find_brightness_threshold_regions(
        profile: TurbidityProfile,
        threshold: float = TURBIDITY_PARAMS.get('brightness_threshold', 0.5),
        min_region_size: int = TURBIDITY_PARAMS.get('min_region_size', 10)
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Finds regions where brightness exceeds threshold
    and returns detailed statistics for each region.

    Args:
        profile: TurbidityProfile object
        threshold: Brightness threshold (0-1)
        min_region_size: Minimum region size in pixels

    Returns:
        Tuple of (crossing_points, region_details)
        - crossing_points: List of normalized y-positions where threshold is first crossed
        - region_details: List of dicts with detailed stats for each threshold-exceeding region
    """
    norm = profile.normalized_profile
    excluded = profile.excluded_regions

    # analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    region_norm = norm[start_idx:end_idx]

    if len(region_norm) == 0:
        return [], []

    # map indices to normalized height
    full_height = excluded['analysis_height']
    y_positions = np.linspace(0, 1, full_height)
    region_y = y_positions[start_idx:end_idx]

    # where brightness exceeds threshold
    above_threshold = region_norm > threshold

    # crossing points
    crossing_points = []
    crossed = False
    for i, (val, y) in enumerate(zip(region_norm, region_y)):
        if not crossed and val > threshold:
            crossing_points.append(float(y))
            crossed = True

    # contiguous regions above threshold
    regions = []
    in_region = False
    start_idx_local = None

    for i, (val, y) in enumerate(zip(above_threshold, region_y)):
        if val and not in_region:
            start_idx_local = i
            in_region = True
        elif not val and in_region:
            # end of region
            end_idx_local = i
            region_size = end_idx_local - start_idx_local

            if region_size >= min_region_size:
                # statistics for region
                region_data = region_norm[start_idx_local:end_idx_local]

                regions.append({
                    # 'start_normalized': float(region_y[start_idx_local]),
                    # 'end_normalized': float(region_y[end_idx_local - 1]),
                    'start_normalized': normalize_to_full_image(start_idx + start_idx_local, profile),
                    'end_normalized': normalize_to_full_image(start_idx + end_idx_local, profile),
                    'center_normalized': float((region_y[start_idx_local] + region_y[end_idx_local - 1]) / 2),
                    'start_absolute': int(start_idx + start_idx_local),
                    'end_absolute': int(start_idx + end_idx_local),
                    'height_pixels': int(region_size),
                    'mean_brightness': float(np.mean(region_data)),
                    'std_brightness': float(np.std(region_data)),
                    'variance': float(np.var(region_data)),
                    'min_brightness': float(np.min(region_data)),
                    'max_brightness': float(np.max(region_data)),
                    'peak_brightness': float(np.max(region_data)),
                    'threshold': threshold,
                    'raw_values': region_data.tolist()
                })

            in_region = False

    # handle case where region extends to end
    if in_region and len(region_y) - start_idx_local >= min_region_size:
        region_data = region_norm[start_idx_local:]

        regions.append({
            'start_normalized': float(region_y[start_idx_local]),
            'end_normalized': float(region_y[-1]),
            'center_normalized': float((region_y[start_idx_local] + region_y[-1]) / 2),
            'start_absolute': int(start_idx + start_idx_local),
            'end_absolute': int(start_idx + len(region_norm)),
            'height_pixels': int(len(region_norm) - start_idx_local),
            'mean_brightness': float(np.mean(region_data)),
            'std_brightness': float(np.std(region_data)),
            'variance': float(np.var(region_data)),
            'min_brightness': float(np.min(region_data)),
            'max_brightness': float(np.max(region_data)),
            'peak_brightness': float(np.max(region_data)),
            'threshold': threshold,
            'raw_values': region_data.tolist()
        })

    return crossing_points, regions


def _find_brightness_change_points(
        smoothed: np.ndarray,
        change_threshold: float,
        min_distance: int,
        window: int,
) -> np.ndarray:
    """
    Find vertical positions where avg brightness changes sharply.

    For each candidate position i we compare the mean brightness of a small
    window above and a small window below i. If the absolute difference
    between these two means exceeds change_threshold, i is a change-point.
    """
    n = len(smoothed)
    if n < 2 * window + 3:
        return np.array([], dtype=int)

    window = max(1, min(window, n // 4))

    change_score = np.zeros(n, dtype=float)

    for i in range(window, n - window):
        before = smoothed[i - window:i]
        after = smoothed[i:i + window]
        change_score[i] = abs(after.mean() - before.mean())

    peaks, _ = find_peaks(
        change_score,
        height=change_threshold,
        distance=min_distance
    )
    return peaks


def detect_sudden_brightness_changes(
        profile: TurbidityProfile,
        min_intensity_change: float,
        min_span_fraction: float,
        max_span_fraction: float,
        smoothing_sigma: float,
        gradient_epsilon: float,
) -> List[Dict[str, Any]]:
    """
    Detect sudden brightness changes
    Each event is a boundary between two neighbouring plateaus/layers.
    """
    norm = profile.normalized_profile
    excluded = profile.excluded_regions

    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    full_height = excluded['analysis_height']

    analysis_region = norm[start_idx:end_idx]
    if len(analysis_region) == 0:
        return []

    smoothed = gaussian_filter1d(analysis_region, sigma=smoothing_sigma)

    N = len(analysis_region)
    # sudden brightness change boundaries
    min_span_pixels = max(1, int(N * min_span_fraction))
    max_span_pixels = max(min_span_pixels, int(N * max_span_fraction))
    window = max(3, min_span_pixels // 2)

    change_points = _find_brightness_change_points(
        smoothed=smoothed,
        change_threshold=min_intensity_change,
        min_distance=min_span_pixels,
        window=window,
    )

    events: List[Dict[str, Any]] = []

    for cp in change_points:
        # left and right short segments only used for stats
        left_idx = max(0, cp - window)
        right_idx = min(N, cp + window)

        left_mean = float(np.mean(smoothed[left_idx:cp]))
        right_mean = float(np.mean(smoothed[cp:right_idx]))
        intensity_change = right_mean - left_mean

        span_pixels = right_idx - left_idx
        if span_pixels < min_span_pixels or span_pixels > max_span_pixels:
            continue

        direction = "increasing" if intensity_change > 0 else "decreasing"

        start_abs = start_idx + left_idx
        end_abs = start_idx + right_idx

        events.append({
            "center_index": int(cp),
            "start_index": int(left_idx),
            "end_index": int(right_idx),
            "start_absolute": int(start_abs),
            "end_absolute": int(end_abs),
            # "start_norm": float(start_abs / full_height),
            # "end_norm": float(end_abs / full_height),
            "start_norm": normalize_to_full_image(start_abs, profile),
            "end_norm": normalize_to_full_image(end_abs, profile),
            "direction": direction,
            "intensity_change": float(intensity_change),
            "span_pixels": int(span_pixels),
            "span_fraction": float(span_pixels / N),
        })

    return events
