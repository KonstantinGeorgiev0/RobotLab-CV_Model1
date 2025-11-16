"""
Turbidity profile analysis for phase separation detection.
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
from visualization.turbidity_viz import save_enhanced_turbidity_plot


@dataclass
class TurbidityProfile:
    """Container for turbidity analysis results."""
    raw_profile: np.ndarray
    normalized_profile: np.ndarray
    excluded_regions: Dict[str, Any]
    gradient: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    centerline_x: Optional[int] = None


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


def detect_turbidity_peaks(profile: TurbidityProfile) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect significant peaks in turbidity profile indicating phase separation.

    Args:
        profile: TurbidityProfile object

    Returns:
        Tuple of (is_phase_separated, analysis_metrics)
    """
    norm_profile = profile.normalized_profile
    excluded = profile.excluded_regions

    # Extract only the analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    analysis_region = norm_profile[start_idx:end_idx]

    if len(analysis_region) < 50:  # Minimum profile length
        return False, {'reason': 'profile_too_short'}

    # Calculate gradient
    gradient = np.abs(np.gradient(analysis_region))

    # Dynamic threshold calculation
    mean_grad = np.mean(gradient)
    std_grad = np.std(gradient)
    threshold = max(
        mean_grad + TURBIDITY_PARAMS['gradient_threshold_sigma'] * std_grad,
        TURBIDITY_PARAMS['gradient_threshold_min']
    )

    # Find peaks above threshold
    peak_indices = np.where(gradient > threshold)[0]

    if len(peak_indices) == 0:
        return False, {'reason': 'no_peaks', 'threshold': threshold}

    # Group nearby peaks
    min_separation = int(len(analysis_region) * TURBIDITY_PARAMS['peak_separation_fraction'])
    peak_groups = []
    current_group = [peak_indices[0]]

    for i in range(1, len(peak_indices)):
        if peak_indices[i] - peak_indices[i-1] < min_separation:
            current_group.append(peak_indices[i])
        else:
            peak_groups.append(current_group)
            current_group = [peak_indices[i]]

    if current_group:
        peak_groups.append(current_group)

    # Select strongest peak from each group
    merged_peaks = []
    for group in peak_groups:
        strongest_idx = group[np.argmax(gradient[group])]
        merged_peaks.append(strongest_idx)

    # Store results in profile
    profile.gradient = gradient
    profile.peaks = np.array(merged_peaks)

    # Determine if phase separated
    is_separated = len(merged_peaks) >= 2

    metrics = {
        'num_peaks': len(merged_peaks),
        'peak_positions': merged_peaks,
        'threshold': threshold,
        'mean_gradient': mean_grad,
        'std_gradient': std_grad,
        'max_gradient': np.max(gradient) if len(gradient) > 0 else 0
    }

    return is_separated, metrics


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

    print(width)
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


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="vial_image.jpg")
    parser.add_argument("-o", "--out_dir", type=str, default="output")
    parser.add_argument("-b", "--brightness_threshold", type=float, default=TURBIDITY_PARAMS['brightness_threshold'])
    parser.add_argument("-c", "--centerline_width", type=int, default=TURBIDITY_PARAMS['centerline_width'])
    parser.add_argument('--gradient-threshold', type=float, default=0.05, help='Gradient threshold for transitions')
    args = parser.parse_args()
    image_path = args.image_path
    out_dir = args.out_dir
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

    # find where brightness
    bright_crossings, bright_regions = find_brightness_threshold_regions(
        turbidity_profile,
        threshold=TURBIDITY_PARAMS['brightness_threshold']
    )

    regions = segment_brightness_regions(centerline_profile)
    crossings, brightness_regions = find_brightness_threshold_regions(
        centerline_profile, threshold=TURBIDITY_PARAMS['brightness_threshold']
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

    # resized analysis image for region stats
    # analyse_img = preprocess_img(img)['analysis_image']

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
    plot_path = save_enhanced_turbidity_plot(
        image_path,
        turbidity_profile.normalized_profile,
        getattr(turbidity_profile, "excluded_regions", None),
        out_dir
    )
    # center_plot_path = save_enhanced_turbidity_plot(
    #     image_path,
    #     centerline_profile.normalized_profile,
    #     getattr(centerline_profile, "excluded_regions", None),
    #     out_dir
    # )

    # turbidity statistics
    results = {
        "mean": float(np.mean(turbidity_profile.normalized_profile)),
        "std": float(np.std(turbidity_profile.normalized_profile)),
        "max_gradient": float(np.max(np.abs(np.gradient(turbidity_profile.normalized_profile))))
        if len(turbidity_profile.normalized_profile) > 1 else 0.0
    }

    print(f"Turbidity statistics: {results}")
    print(f"\n=== Centerline Turbidity Analysis ===")
    print(f"Centerline position: x={centerline_profile.centerline_x}")
    print(f"Centerline width: {centerline_profile.excluded_regions['centerline_width']} pixels")
    print(f"\n=== Brightness Analysis ===")
    print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} at normalized height(s): {bright_crossings}")
    print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} "
          f"in contiguous {len(bright_regions)} regions: {bright_regions}")
    print(f"\n=== Features ===")
    print(f"\nNumber of regions: {len(features)}")
    print(f"\nFeatures: ")
    print('\n\n'.join(
        f"Region {i}:\n" + '\n'.join(f"  {k}: {v}" for k, v in item.items()) for i, item in enumerate(features)))
    # print(f"\nRegion features: '\n'.join(str(feature) for feature in features)")
    # print(f"Analysis image shape: {analysis_img.shape}")
    # print(f"Exclusion zones: top={ex['excluded_top_fraction']:.2f}, "
    #       f"bottom={ex['excluded_bottom_fraction']:.2f}, "
    #       f"left={ex['excluded_left_fraction']:.2f}, right={ex['excluded_right_fraction']:.2f}")
    # print(f"Region stats: {stats}")