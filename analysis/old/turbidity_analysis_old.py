"""
Turbidity profile analysis for phase separation detection.
TODO: Split file into segments - turbidity_preprocessing,
    turbidity_profiles, turbidity_segmentation, turbidity_stats, turbidity_cli, turbidity_decision
TODO: get object from preprocess result
TODO: remove magic numbers
TODO: add file to extract data for classification logic turbidity_decision
"""
import argparse
import sys
from pathlib import Path

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
# from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import TURBIDITY_PARAMS
from visualization.turbidity_viz import save_turbidity_plot


@dataclass
class TurbidityProfile:
    """Container for turbidity analysis results."""
    raw_profile: np.ndarray
    normalized_profile: np.ndarray
    excluded_regions: Dict[str, Any]
    gradient: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    centerline_x: Optional[int] = None


@dataclass
class PreprocessResult:
    original_image: np.ndarray
    original_height: int
    original_width: int
    analysis_image: np.ndarray
    height: int
    width: int
    gray: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    raw_profile: np.ndarray


def preprocess_img(image: np.ndarray) -> Dict[str, Any]:
    h_original, w_original = image.shape[:2]
    # Resize to analysis dimensions
    analysis_img = cv2.resize(image,
                              (TURBIDITY_PARAMS['analysis_width'],
                               TURBIDITY_PARAMS['analysis_height']))

    height, width = analysis_img.shape[:2]
    # different color spaces
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)
    # row wise mean intensity
    raw_profile = np.mean(gray, axis=1)

    return {
        'original_image': image,
        'original_height': h_original,
        'original_width': w_original,
        'analysis_image': analysis_img,
        'height': height,
        'width': width,
        'gray': gray,
        'hsv': hsv,
        'lab': lab,
        'raw_profile': raw_profile,
    }


def compute_turbidity_profile(image: np.ndarray,
                             cap_bottom_y: Optional[int] = None) -> TurbidityProfile:
    """
    Compute turbidity profile with smart region exclusion.

    Args:
        image: BGR image array
        cap_bottom_y: Y-coordinate of cap bottom (if detected)

    Returns:
        TurbidityProfile object with analysis results
    """
    (img_original, h_original, w_original,
     analysis_img, height, width, gray, hsv, lab , raw_profile) = preprocess_img(image).values()
    # TODO: return object from preprocess image, not like this

    # Multiple channel analysis
    saturation = hsv[:, :, 1]  # Saturation channel
    lightness = lab[:, :, 0]  # Lightness channel

    # Compute multiple profiles
    profiles = {
        'intensity': np.mean(gray, axis=1),
        'saturation': np.mean(saturation, axis=1),
        'lightness': np.mean(lightness, axis=1),
        'variance': np.var(gray, axis=1)  # Texture/variance profile
    }

    # Enhanced gradient computation
    gradients = {}
    for name, profile in profiles.items():
        # First derivative (gradient)
        grad1 = np.gradient(profile)
        # Second derivative (curvature)
        grad2 = np.gradient(grad1)
        gradients[name] = {
            'first_derivative': grad1,
            'second_derivative': grad2,
            'absolute_gradient': np.abs(grad1)
        }

    # exclusion regions
    if cap_bottom_y is not None:
        # scale cap position
        cap_bottom_scaled = int((cap_bottom_y / h_original) * TURBIDITY_PARAMS['analysis_height'])
        # buffer below cap
        top_exclude_idx = min(cap_bottom_scaled + 10,
                             TURBIDITY_PARAMS['analysis_height'] // 3)
    else:
        # default top exclusion
        top_exclude_idx = int(height *
                             TURBIDITY_PARAMS['top_exclude_fraction'])

    # bottom exclusion
    bottom_exclude_idx = int(height *
                            (1 - TURBIDITY_PARAMS['bottom_exclude_fraction']))

    # horizontal exclusions
    left_exclude_idx = int(width * TURBIDITY_PARAMS['left_exclude_fraction'])
    right_exclude_idx = int(width * (1 - TURBIDITY_PARAMS['right_exclude_fraction']))

    # print("\nLeft exclusion: ", left_exclude_idx)
    # print("\nRight exclusion: ", right_exclude_idx)
    # print("\nTURBIDITY PARAMS LEFT RIGHT: ", TURBIDITY_PARAMS['left_exclude_fraction'], TURBIDITY_PARAMS['right_exclude_fraction'])

    # horizontal exclusion to channel data
    gray = gray[:, left_exclude_idx:right_exclude_idx]
    saturation = saturation[:, left_exclude_idx:right_exclude_idx]
    lightness = lightness[:, left_exclude_idx:right_exclude_idx]

    # recompute profiles
    profiles = {
        'intensity': np.mean(gray, axis=1),
        'saturation': np.mean(saturation, axis=1),
        'lightness': np.mean(lightness, axis=1),
        'variance': np.var(gray, axis=1)
    }

    # update profile with horizontally excluded intensity
    raw_profile = profiles['intensity']

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

    # full normalized profile with excluded regions marked
    full_normalized = np.zeros_like(raw_profile)
    if len(normalized) > 0:
        full_normalized[top_exclude_idx:bottom_exclude_idx] = normalized

    excluded_info = {
        'top_exclude_idx': top_exclude_idx,
        'bottom_exclude_idx': bottom_exclude_idx,
        'analysis_height': TURBIDITY_PARAMS['analysis_height'],
        'excluded_top_fraction': top_exclude_idx / TURBIDITY_PARAMS['analysis_height'],
        'excluded_bottom_fraction': (TURBIDITY_PARAMS['analysis_height'] - bottom_exclude_idx) /
                                   TURBIDITY_PARAMS['analysis_height'],
        'cap_detected': cap_bottom_y is not None,
        'left_exclude_idx': left_exclude_idx,
        'right_exclude_idx': right_exclude_idx,
        'excluded_left_fraction': left_exclude_idx / TURBIDITY_PARAMS['analysis_width'],
        'excluded_right_fraction': (TURBIDITY_PARAMS['analysis_width'] - right_exclude_idx) / TURBIDITY_PARAMS[
            'analysis_width'],
    }

    return TurbidityProfile(
        raw_profile=raw_profile,
        normalized_profile=full_normalized,
        excluded_regions=excluded_info
    )


# def detect_turbidity_peaks(profile: TurbidityProfile) -> Tuple[bool, Dict[str, Any]]:
#     """
#     Detect significant peaks in turbidity profile indicating phase separation.
#
#     Args:
#         profile: TurbidityProfile object
#
#     Returns:
#         Tuple of (is_phase_separated, analysis_metrics)
#     """
#     norm_profile = profile.normalized_profile
#     excluded = profile.excluded_regions
#
#     # Extract only the analysis region
#     start_idx = excluded['top_exclude_idx']
#     end_idx = excluded['bottom_exclude_idx']
#     analysis_region = norm_profile[start_idx:end_idx]
#
#     if len(analysis_region) < 50:  # Minimum profile length
#         return False, {'reason': 'profile_too_short'}
#
#     # Calculate gradient
#     gradient = np.abs(np.gradient(analysis_region))
#
#     # Dynamic threshold calculation
#     mean_grad = np.mean(gradient)
#     std_grad = np.std(gradient)
#     threshold = max(
#         mean_grad + TURBIDITY_PARAMS['gradient_threshold_sigma'] * std_grad,
#         TURBIDITY_PARAMS['gradient_threshold_min']
#     )
#
#     # Find peaks above threshold
#     peak_indices = np.where(gradient > threshold)[0]
#
#     if len(peak_indices) == 0:
#         return False, {'reason': 'no_peaks', 'threshold': threshold}
#
#     # Group nearby peaks
#     min_separation = int(len(analysis_region) * TURBIDITY_PARAMS['peak_separation_fraction'])
#     peak_groups = []
#     current_group = [peak_indices[0]]
#
#     for i in range(1, len(peak_indices)):
#         if peak_indices[i] - peak_indices[i-1] < min_separation:
#             current_group.append(peak_indices[i])
#         else:
#             peak_groups.append(current_group)
#             current_group = [peak_indices[i]]
#
#     if current_group:
#         peak_groups.append(current_group)
#
#     # Select strongest peak from each group
#     merged_peaks = []
#     for group in peak_groups:
#         strongest_idx = group[np.argmax(gradient[group])]
#         merged_peaks.append(strongest_idx)
#
#     # Store results in profile
#     profile.gradient = gradient
#     profile.peaks = np.array(merged_peaks)
#
#     # Determine if phase separated
#     is_separated = len(merged_peaks) >= 2
#
#     metrics = {
#         'num_peaks': len(merged_peaks),
#         'peak_positions': merged_peaks,
#         'threshold': threshold,
#         'mean_gradient': mean_grad,
#         'std_gradient': std_grad,
#         'max_gradient': np.max(gradient) if len(gradient) > 0 else 0
#     }
#
#     return is_separated, metrics


def analyze_region_turbidity(
    region_image: np.ndarray,
    top_frac: float = 0.0,
    bottom_frac: float = 0.0,
    left_frac: float = 0.0,
    right_frac: float = 0.0
) -> Dict[str, float]:
    """
    Analyze turbidity in a region with configurable exclusions.
    """
    if region_image.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'max_gradient': 0.0}

    h, w = region_image.shape[:2]

    # Apply exclusions
    top_idx = int(h * top_frac)
    bottom_idx = int(h * (1 - bottom_frac))
    left_idx = int(w * left_frac)
    right_idx = int(w * (1 - right_frac))

    # Crop with exclusions
    cropped = region_image[top_idx:bottom_idx, left_idx:right_idx]
    if cropped.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'max_gradient': 0.0}

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

    # Row-wise means
    row_means = np.mean(gray, axis=1)
    if row_means.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'max_gradient': 0.0}

    # Normalize
    min_val, max_val = row_means.min(), row_means.max()
    if max_val > min_val:
        normalized = (row_means - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(row_means)

    # Stats
    stats = {
        'mean': float(np.mean(normalized)),
        'std': float(np.std(normalized)),
        'max_gradient': float(np.max(np.abs(np.gradient(normalized)))) if len(normalized) > 1 else 0.0
    }
    return stats


def compute_centerline_turbidity_profile(
        image: np.ndarray,
        centerline_width: int = TURBIDITY_PARAMS['centerline_width'],
        top_exclude_fraction: float = TURBIDITY_PARAMS['top_exclude_fraction'],
        bottom_exclude_fraction: float = TURBIDITY_PARAMS['bottom_exclude_fraction']
):
    (img_original, h_original, w_original,
     analysis_img, height, width, gray, hsv, lab, raw_profile) = preprocess_img(image).values()
    # TODO: return object from preprocess image, not like this

    # determine horiz center
    center_x = int(width / 2)
    half_width = centerline_width // 2
    # centerline region
    left_bound = center_x - half_width
    right_bound = center_x + half_width
    centerline_region = analysis_img[:, left_bound:right_bound]

    # channels
    gray, hsv, lab = (preprocess_img(centerline_region)['gray'],
                      preprocess_img(centerline_region)['hsv'],
                      preprocess_img(centerline_region)['lab'])

    # profiles
    profile = {
        'intensity': np.mean(gray, axis=1),
        'saturation': np.mean(hsv[:, :, 1], axis=1),
        'lightness': np.mean(lab[:, :, 0], axis=1),
        'variance': np.var(gray, axis=1)
    }

    raw_profile = profile['intensity']

    # exclusions
    top_exclusion_idx = int(height * top_exclude_fraction)
    bottom_exclusion_idx = int(height * (1 - bottom_exclude_fraction))

    analysis_region = raw_profile[top_exclusion_idx:bottom_exclusion_idx]

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

    # full normalized profile
    full_normalized = np.zeros_like(raw_profile)
    if len(normalized) > 0:
        full_normalized[top_exclusion_idx:bottom_exclusion_idx] = normalized

    excluded_info = {
        'top_exclude_idx': top_exclusion_idx,
        'bottom_exclude_idx': bottom_exclusion_idx,
        'analysis_height': TURBIDITY_PARAMS['analysis_height'],
        'excluded_top_fraction': top_exclusion_idx / TURBIDITY_PARAMS['analysis_height'],
        'excluded_bottom_fraction': (TURBIDITY_PARAMS['analysis_height'] - bottom_exclusion_idx) / TURBIDITY_PARAMS['analysis_height'],
        'centerline_x': center_x,
        'centerline_width': centerline_width,
        'centerline_bounds': (left_bound, right_bound)
    }

    return TurbidityProfile(
        raw_profile=raw_profile,
        normalized_profile=full_normalized,
        excluded_regions=excluded_info,
        centerline_x=center_x
    )


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

    # Find boundary points (peaks in gradient)
    boundary_peaks, _ = find_peaks(abs_gradient, height=gradient_threshold, distance=10)

    # Create segments between boundaries
    # Start with full region, then split at boundaries
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
        min_intensity_change: float = 0.2,
        min_span_fraction: float = 0.01,
        max_span_fraction: float = 0.10,
        smoothing_sigma: float = 1.0,
        gradient_epsilon: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Detect sudden brightness changes over a short vertical span.

    A "sudden change" is defined as a mostly-monotonic segment of the
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

        # Absolute pixel indices in analysis-height coordinates
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

    # Walk the gradient and build monotonic segments
    for i, g in enumerate(grad):
        if abs(g) < gradient_epsilon:
            # we're basically flat here: close segment if open
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


def detect_phase_separation_from_separations(
        separation_events: List[Dict[str, Any]],
        min_liquid_interfaces: int = 1,
        min_vertical_span: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide if the vial is phase-separated based on separation events.

    A 'liquid interface' is any boundary between two liquid phases:
      - opaque-translucent
      - translucent-opaque
      - liquid-liquid

    We can optionally require that the interfaces span some vertical range
    to avoid tiny noisy splits.
    """
    if not separation_events:
        return False, {"reason": "no_separation_events"}

    liquid_types = {"opaque-translucent", "translucent-opaque", "liquid-liquid"}

    # collect positions of all liquid-liquid-type interfaces
    liquid_ifaces = [
        ev for ev in separation_events
        if ev["type"] in liquid_types
    ]

    if len(liquid_ifaces) < min_liquid_interfaces:
        return False, {
            "reason": "too_few_liquid_interfaces",
            "num_liquid_interfaces": len(liquid_ifaces)
        }

    # measure vertical extent of those interfaces in normalized coordinates
    ys = [ev["boundary_norm"] for ev in liquid_ifaces if "boundary_norm" in ev]
    if ys:
        span = max(ys) - min(ys)
    else:
        span = 0.0

    if span < min_vertical_span:
        return False, {
            "reason": "liquid_interfaces_span_too_small",
            "num_liquid_interfaces": len(liquid_ifaces),
            "span": span,
        }

    return True, {
        "reason": "ok",
        "num_liquid_interfaces": len(liquid_ifaces),
        "span": span,
        "interfaces": liquid_ifaces,
    }


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

    # analysis region only (exclude top/bottom caps)
    analysis_region = norm[top_idx:bottom_idx]
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
            "height_pixels": int(e - s),
            "mean_brightness": mean,
            "variance": var,
            "std_brightness": std,
        })

    return {
        "overall_variance": overall_var,
        "segments": segments,
    }


def classify_segment_phase(mean_brightness: float) -> str:
    """
    Classify a segment into AIR / LIQUID_TRANSLUCENT / LIQUID_OPAQUE
    based purely on mean brightness (0–1).
    Adjust thresholds in PHASE_THRESHOLDS to your dataset.
    """
    if mean_brightness <= TURBIDITY_PARAMS["air_max"]:
        return "AIR"
    elif mean_brightness <= TURBIDITY_PARAMS["translucent_max"]:
        return "LIQUID_TRANSLUCENT"
    else:
        return "LIQUID_OPAQUE"


def label_segments(
        segments: List[Dict[str, Any]],
        analysis_height: int
) -> List[Dict[str, Any]]:
    """
    Add phase labels + height fraction to each segment, enforcing:
      - air can only appear above the first liquid layer
      - any segment below a liquid that would otherwise be AIR
        is treated as LIQUID_TRANSLUCENT
    """
    labeled: List[Dict[str, Any]] = []
    seen_liquid = False  # have we passed the first liquid layer?

    for seg in segments:  # segments must be sorted top->bottom
        h_frac = seg["height_pixels"] / analysis_height
        if h_frac < TURBIDITY_PARAMS["min_segment_height_frac"]:
            # ignore very thin noisy segments
            continue

        phase = classify_segment_phase(seg["mean_brightness"])

        # once we've seen a liquid, never allow air below it
        if seen_liquid and phase == "AIR":
            phase = "LIQUID_TRANSLUCENT"

        if phase.startswith("LIQUID"):
            seen_liquid = True

        seg_labeled = {
            **seg,
            "phase": phase,
            "height_frac": h_frac,
        }
        labeled.append(seg_labeled)

    return labeled



def detect_separation_types(
        segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Given labeled segments (with 'phase', 'mean_brightness', 'start_norm', 'end_norm'),
    return a list of separation events with type:
      - 'air-liquid'
      - 'liquid-liquid'
      - 'opaque-translucent'
      - 'translucent-opaque'
    """
    events = []
    if len(segments) < 2:
        return events

    contrast_min = TURBIDITY_PARAMS["liquid_liquid_contrast"]

    for upper, lower in zip(segments, segments[1:]):
        p_top = upper["phase"]
        p_bottom = lower["phase"]
        mu_top = upper["mean_brightness"]
        mu_bottom = lower["mean_brightness"]
        delta_mu = abs(mu_bottom - mu_top)

        # Skip very small contrast differences
        if delta_mu < contrast_min:
            continue

        # Determine separation type
        if "AIR" in (p_top, p_bottom) and (p_top != p_bottom):
            sep_type = "air-liquid"

        elif {p_top, p_bottom} == {"LIQUID_OPAQUE", "LIQUID_TRANSLUCENT"}:
            if p_top == "LIQUID_OPAQUE" and p_bottom == "LIQUID_TRANSLUCENT":
                sep_type = "opaque-translucent"
            else:
                sep_type = "translucent-opaque"

        elif p_top.startswith("LIQUID") and p_bottom.startswith("LIQUID"):
            # both liquids, different brightness enough
            sep_type = "liquid-liquid"
        else:
            # same phase or something weird – ignore
            continue

        events.append({
            "type": sep_type,
            "top_phase": p_top,
            "bottom_phase": p_bottom,
            "delta_brightness": float(delta_mu),
            "boundary_norm": float(upper["end_normalized"]),  # y-position of interface
            "top_segment": upper,
            "bottom_segment": lower,
        })

    return events


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="vial_image.jpg")
    parser.add_argument("-o", "--out_dir", type=str, default="output")
    parser.add_argument("-b", "--brightness_threshold", type=float, default=TURBIDITY_PARAMS['brightness_threshold'])
    parser.add_argument("-c", "--centerline_width", type=int, default=TURBIDITY_PARAMS['centerline_width'])
    parser.add_argument("--gradient-threshold", type=float, default=0.05, help='Gradient threshold for transitions')
    parser.add_argument(
        '--min-change',
        type=float,
        default=0.2,
        help='Minimum normalized intensity change ΔI for a sudden transition'
    )
    parser.add_argument(
        '--min-span-frac',
        type=float,
        default=0.01,
        help='Minimum vertical span (fraction of analysis height) to ignore noise'
    )
    parser.add_argument(
        '--max-span-frac',
        type=float,
        default=0.10,
        help='Maximum vertical span (fraction of analysis height) to still be considered sudden'
    )
    parser.add_argument(
        '--smooth-sigma',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma for brightness profile before change detection'
    )

    args = parser.parse_args()
    image_path = args.image_path
    out_dir = args.out_dir
    out_dir_center = out_dir + "/center"
    TURBIDITY_PARAMS['centerline_width'] = args.centerline_width
    TURBIDITY_PARAMS['brightness_threshold'] = args.brightness_threshold

    # turbidity analysis if image exists
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {args.image}")
        exit(1)

    turbidity_profile = compute_turbidity_profile(img)
    # centered turbidity profile
    centerline_profile = compute_centerline_turbidity_profile(
        image=img,
        centerline_width=TURBIDITY_PARAMS['centerline_width']
    )

    # sudden brightness changes along vertical axis
    sudden_changes = detect_sudden_brightness_changes(
        centerline_profile,
        min_intensity_change=args.min_change,
        min_span_fraction=args.min_span_frac,
        max_span_fraction=args.max_span_frac,
        smoothing_sigma=args.smooth_sigma,
        gradient_epsilon=args.gradient_threshold
    )

    # compute variance
    variance_stats = compute_variance_between_changes(
        centerline_profile,
        sudden_changes
    )

    # find where brightness crosses threshold on full profile
    bright_crossings, bright_regions = find_brightness_threshold_regions(
        turbidity_profile,
        threshold=TURBIDITY_PARAMS['brightness_threshold']
    )

    regions = segment_brightness_regions(centerline_profile)
    crossings, brightness_regions = find_brightness_threshold_regions(
        centerline_profile, threshold=TURBIDITY_PARAMS['brightness_threshold']
    )

    # distinct separation layers
    analysis_height = centerline_profile.excluded_regions["analysis_height"]
    # labeled_segments = label_segments(regions, analysis_height)
    segments = variance_stats["segments"]
    labeled_segments = label_segments(segments, analysis_height)
    separation_events = detect_separation_types(labeled_segments)

    # detect phase separation pattern
    phase_sep, phase_info = detect_phase_separation_from_separations(
        separation_events,
        min_liquid_interfaces=2,
        min_vertical_span=0.05
    )

    # get features for all regions
    features = []
    for region in regions:
        features.append({
            'region_id': region['region_id'],
            'start_normalized': float(f"{region['start_normalized']:.2f}"),
            'end_normalized': float(f"{region['end_normalized']:.2f}"),
            'center_normalized': float(f"{region['center_normalized']:.2f}"),
            # 'start_absolute': region['start_absolute'],
            # 'end_absolute': region['end_absolute'],
            'variance': float(f"{region['variance']:.2f}"),
            'height_pixels': region['height_pixels'],
            'mean_brightness': float(f"{region['mean_brightness']:.2f}"),
            'std_brightness': float(f"{region['std_brightness']:.2f}"),
        })

    # extract exclusion fractions
    ex = turbidity_profile.excluded_regions
    stats = analyze_region_turbidity(
        region_image=img,
        top_frac=ex['excluded_top_fraction'],
        bottom_frac=ex['excluded_bottom_fraction'],
        left_frac=ex['excluded_left_fraction'],
        right_frac=ex['excluded_right_fraction']
    )

    # save plot
    plot_path = save_turbidity_plot(
        image_path,
        turbidity_profile.normalized_profile,
        getattr(turbidity_profile, "excluded_regions", None),
        out_dir
    )

    # save plot for centerline profile
    center_plot_path = save_turbidity_plot(
        image_path,
        centerline_profile.normalized_profile,
        getattr(centerline_profile, "excluded_regions", None),
        out_dir,
        change_events=sudden_changes,
        suffix=".centerline_turbidity_enhanced.png"
    )

    # turbidity statistics
    results = {
        "mean": float(np.mean(turbidity_profile.normalized_profile)),
        "std": float(np.std(turbidity_profile.normalized_profile)),
        "max_gradient": float(np.max(np.abs(np.gradient(turbidity_profile.normalized_profile)))),
        "variance": float(np.var(turbidity_profile.normalized_profile))
        if len(turbidity_profile.normalized_profile) > 1 else 0.0
    }

    print(f"Turbidity statistics: {results}")
    print(f"\n=== Centerline Turbidity Analysis ===")
    print(f"Centerline position: x={centerline_profile.centerline_x}")
    print(f"Centerline width: {centerline_profile.excluded_regions['centerline_width']} pixels")

    print(f"\n=== Brightness Analysis ===")
    print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} at normalized height(s): {bright_crossings}")
    print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} "
          f"in contiguous {len(bright_regions)} regions")

    print(f"\n=== Sudden Brightness Changes (centerline) ===")
    print(f"Number of sudden changes: {len(sudden_changes)}")
    for i, ev in enumerate(sudden_changes):
        print(
            f"  Change {i}: "
            f"{ev['direction']} | "
            f"start_norm={ev['start_norm']:.3f}, end_norm={ev['end_norm']:.3f}, "
            f"ΔI={ev['intensity_change']:.3f}, "
            f"span={ev['span_pixels']} px ({ev['span_fraction']:.3f} of analysis height)"
        )

    print(f"\n=== Features ===")
    print(f"\nNumber of regions: {len(features)}")
    print(f"\nFeatures: ")
    print('\n\n'.join(
        f"Region {i}:\n" + '\n'.join(f"  {k}: {v}" for k, v in item.items()) for i, item in enumerate(features)))

    print("\n=== Variance Analysis (centerline, between sudden changes) ===")
    print(f"Overall variance in analysis region: {variance_stats['overall_variance']:.4f}")
    print(f"Number of sudden changes: {len(sudden_changes)}")
    print("Segments between changes:")
    for seg in variance_stats["segments"]:
        print(
            f"  Segment {seg['segment_id']}: "
            f"{seg['start_normalized']:.3f} → {seg['end_normalized']:.3f}, "
            f"len={seg['height_pixels']} px, "
            f"mean={seg['mean_brightness']:.3f}, "
            f"var={seg['variance']:.4f}"
        )

    print("\n=== Phase separation decision ===")
    print(f"Phase separated (from separations): {phase_sep}")
    if phase_sep:
        print(
            f"  Liquid interfaces: {phase_info['num_liquid_interfaces']}, "
            f"span={phase_info['span']:.3f}"
        )

    print("\n=== Separation types from brightness profile ===")
    for ev in separation_events:
        print(
            f"  {ev['type']} at y={ev['boundary_norm']:.3f} "
            f"(Δμ={ev['delta_brightness']:.3f}, "
            f"{ev['top_phase']} → {ev['bottom_phase']})"
        )

    # print(f"\nRegion features: '\n'.join(str(feature) for feature in features)")
    # print(f"Analysis image shape: {analysis_img.shape}")
    # print(f"Exclusion zones: top={ex['excluded_top_fraction']:.2f}, "
    #       f"bottom={ex['excluded_bottom_fraction']:.2f}, "
    #       f"left={ex['excluded_left_fraction']:.2f}, right={ex['excluded_right_fraction']:.2f}")
    # print(f"Region stats: {stats}")