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
from analysis.turbidity.turbidity_profiles import TurbidityProfile


def segment_brightness_regions(
        profile: TurbidityProfile,
        similarity_threshold: float = 0.15,
        min_region_size: int = 20,
        smoothing_sigma: float = 2.0,
        gradient_threshold: float = 0.03
) -> List[Dict[str, Any]]:
    """
    Segment the turbidity profile into regions of similar brightness.
    Uses gradient-based boundary detection to split profile into homogeneous segments.

    Args:
        profile: TurbidityProfile object
        similarity_threshold: Maximum std deviation within a region to consider it homogeneous
        min_region_size: Minimum number of pixels for a valid region
        smoothing_sigma: Gaussian smoothing sigma for noise reduction
        gradient_threshold: Threshold for detecting region boundaries (peaks in gradient)

    Returns:
        List of region dictionaries with statistics and position info
    """
    norm_profile = profile.normalized_profile
    excluded = profile.excluded_regions

    # Extract analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    analysis_region = norm_profile[start_idx:end_idx]

    if len(analysis_region) < min_region_size:
        return []

    # Smooth profile to reduce noise
    smoothed = gaussian_filter1d(analysis_region, sigma=smoothing_sigma)

    # Calculate gradient to find boundaries
    gradient = np.gradient(smoothed)
    abs_gradient = np.abs(gradient)

    # Find boundary points
    boundary_peaks, _ = find_peaks(abs_gradient, height=gradient_threshold, distance=10)

    # Create segments between boundaries
    segment_boundaries = [0] + boundary_peaks.tolist() + [len(analysis_region)]

    regions = []

    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]

        # Skip if segment too small
        if seg_end - seg_start < min_region_size:
            continue

        segment_data = smoothed[seg_start:seg_end]
        raw_segment_data = analysis_region[seg_start:seg_end]

        # Calculate statistics
        mean_brightness = np.mean(segment_data)
        std_brightness = np.std(segment_data)
        variance = np.var(segment_data)
        min_brightness = np.min(segment_data)
        max_brightness = np.max(segment_data)

        # Calculate gradient statistics within region
        segment_gradient = abs_gradient[seg_start:seg_end]
        mean_gradient = np.mean(segment_gradient)
        max_gradient = np.max(segment_gradient)

        # Normalized positions
        start_normalized = seg_start / len(analysis_region)
        end_normalized = seg_end / len(analysis_region)
        center_normalized = (start_normalized + end_normalized) / 2

        # Absolute positions in resized image
        start_absolute = start_idx + seg_start
        end_absolute = start_idx + seg_end

        # Height of region
        region_height = seg_end - seg_start

        regions.append({
            'region_id': i,
            'start_normalized': float(start_normalized),
            'end_normalized': float(end_normalized),
            'center_normalized': float(center_normalized),
            'start_absolute': int(start_absolute),
            'end_absolute': int(end_absolute),
            'height_pixels': int(region_height),
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'variance': float(variance),
            'min_brightness': float(min_brightness),
            'max_brightness': float(max_brightness),
            'peak_brightness': float(max_brightness),
            'brightness_range': float(max_brightness - min_brightness),
            'mean_gradient': float(mean_gradient),
            'max_gradient': float(max_gradient),
            'is_homogeneous': std_brightness < similarity_threshold,
            'raw_values': raw_segment_data.tolist(),
            'smoothed_values': segment_data.tolist()
        })

    return regions


def find_brightness_threshold_regions(
        profile: TurbidityProfile,
        threshold: float = TURBIDITY_PARAMS.get('brightness_threshold', 0.5),
        min_region_size: int = 10
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    finds regions where brightness exceeds threshold
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

    # Only consider analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    region_norm = norm[start_idx:end_idx]

    if len(region_norm) == 0:
        return [], []

    # Map indices to normalized height
    full_height = excluded['analysis_height']
    y_positions = np.linspace(0, 1, full_height)
    region_y = y_positions[start_idx:end_idx]

    # Find where brightness exceeds threshold
    above_threshold = region_norm > threshold

    # Find crossing points
    crossing_points = []
    crossed = False
    for i, (val, y) in enumerate(zip(region_norm, region_y)):
        if not crossed and val > threshold:
            crossing_points.append(float(y))
            crossed = True

    # Find contiguous regions above threshold
    regions = []
    in_region = False
    start_idx_local = None

    for i, (val, y) in enumerate(zip(above_threshold, region_y)):
        if val and not in_region:
            start_idx_local = i
            in_region = True
        elif not val and in_region:
            # End of region
            end_idx_local = i
            region_size = end_idx_local - start_idx_local

            if region_size >= min_region_size:
                # Extract statistics for this region
                region_data = region_norm[start_idx_local:end_idx_local]

                regions.append({
                    'start_normalized': float(region_y[start_idx_local]),
                    'end_normalized': float(region_y[end_idx_local - 1]),
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

    # Handle case where region extends to end
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


def detect_sudden_brightness_changes(
        profile: TurbidityProfile,
        min_intensity_change: float = 0.05,
        min_span_fraction: float = 0.01,
        max_span_fraction: float = 0.35,
        smoothing_sigma: float = 0.5,
        gradient_epsilon: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Detect sudden brightness changes over a short vertical span.

    A sudden change is defined as a mostly-monotonic segment of the
    (smoothed) brightness profile where:
      - |ΔI| >= min_intensity_change (normalized units)
      - span_fraction is between [min_span_fraction, max_span_fraction]

    The function returns a list of events with:
      - start/end (absolute indices + normalized 0–1 height)
      - intensity_change
      - span (pixels + fraction)
      - direction: 'increasing' or 'decreasing'
    """
    norm = profile.normalized_profile
    excluded = profile.excluded_regions

    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    full_height = excluded['analysis_height']

    analysis_region = norm[start_idx:end_idx]

    if len(analysis_region) == 0:
        return []

    # Smooth to reduce noise
    smoothed = gaussian_filter1d(analysis_region, sigma=smoothing_sigma)

    # Gradient of smoothed profile
    grad = np.gradient(smoothed)
    abs_grad = np.abs(grad)

    N = len(analysis_region)
    if N < 3:
        return []

    min_span_pixels = max(1, int(N * min_span_fraction))
    max_span_pixels = max(min_span_pixels, int(N * max_span_fraction))

    events: List[Dict[str, Any]] = []

    in_segment = False
    seg_start = 0
    seg_sign = 0

    def close_segment(seg_start_idx: int, seg_end_idx: int, sign: int):
        """Evaluate candidate segment and add to events if it qualifies."""
        span = seg_end_idx - seg_start_idx
        if span <= 0:
            return

        if span < min_span_pixels or span > max_span_pixels:
            return

        start_val = smoothed[seg_start_idx]
        end_val = smoothed[seg_end_idx]
        intensity_change = end_val - start_val

        if abs(intensity_change) < min_intensity_change:
            return

        direction = "increasing" if intensity_change > 0 else "decreasing"

        # absolute pixel indices
        start_abs = start_idx + seg_start_idx
        end_abs = start_idx + seg_end_idx

        start_norm = start_abs / full_height
        end_norm = end_abs / full_height

        segment_grad = abs_grad[seg_start_idx:seg_end_idx + 1]
        mean_grad = float(np.mean(segment_grad))
        max_grad = float(np.max(segment_grad))

        events.append({
            "start_index": int(seg_start_idx),
            "end_index": int(seg_end_idx),
            "start_absolute": int(start_abs),
            "end_absolute": int(end_abs),
            "start_norm": float(start_norm),
            "end_norm": float(end_norm),
            "direction": direction,
            "intensity_change": float(intensity_change),
            "span_pixels": int(span),
            "span_fraction": float(span / N),
            "mean_gradient": mean_grad,
            "max_gradient": max_grad,
        })

    # walk gradient and build monotonic segments
    for i, g in enumerate(grad):
        if abs(g) < gradient_epsilon:
            # close segment if open
            if in_segment:
                close_segment(seg_start, max(i - 1, seg_start), seg_sign)
                in_segment = False
                seg_sign = 0
            continue

        sign = 1 if g > 0 else -1

        if not in_segment:
            in_segment = True
            seg_start = i
            seg_sign = sign
        else:
            # sign change: close previous, start new
            if sign != seg_sign:
                close_segment(seg_start, max(i - 1, seg_start), seg_sign)
                in_segment = True
                seg_start = i
                seg_sign = sign

    # Close any open segment at the end
    if in_segment:
        close_segment(seg_start, N - 1, seg_sign)

    return events
