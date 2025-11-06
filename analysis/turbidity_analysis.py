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
from scipy.stats import linregress

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
    h_original, w_original = image.shape[:2]

    # Resize to standard analysis dimensions
    analysis_img = cv2.resize(image,
                              (TURBIDITY_PARAMS['analysis_width'],
                               TURBIDITY_PARAMS['analysis_height']))

    height, width = analysis_img.shape[:2]

    # Convert to grayscale for turbidity analysis
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)

    # Compute row-wise mean intensity
    raw_profile = np.mean(gray, axis=1)

    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)

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

    # Determine exclusion regions
    if cap_bottom_y is not None:
        # Scale cap position to analysis dimensions
        cap_bottom_scaled = int((cap_bottom_y / h_original) * TURBIDITY_PARAMS['analysis_height'])
        # Add buffer below cap
        top_exclude_idx = min(cap_bottom_scaled + 10,
                             TURBIDITY_PARAMS['analysis_height'] // 3)
    else:
        # Use default top exclusion
        top_exclude_idx = int(height *
                             TURBIDITY_PARAMS['top_exclude_fraction'])

    # Bottom exclusion for vial bottom artifacts
    bottom_exclude_idx = int(height *
                            (1 - TURBIDITY_PARAMS['bottom_exclude_fraction']))

    # Horizontal exclusions
    left_exclude_idx = int(width * TURBIDITY_PARAMS['left_exclude_fraction'])
    right_exclude_idx = int(width * (1 - TURBIDITY_PARAMS['right_exclude_fraction']))

    # print("\nLeft exclusion: ", left_exclude_idx)
    # print("\nRight exclusion: ", right_exclude_idx)
    # print("\nTURBIDITY PARAMS LEFT RIGHT: ", TURBIDITY_PARAMS['left_exclude_fraction'], TURBIDITY_PARAMS['right_exclude_fraction'])

    # Apply horizontal exclusion to all channel data
    gray = gray[:, left_exclude_idx:right_exclude_idx]
    saturation = saturation[:, left_exclude_idx:right_exclude_idx]
    lightness = lightness[:, left_exclude_idx:right_exclude_idx]

    # Recompute profiles using horizontally cropped regions
    profiles = {
        'intensity': np.mean(gray, axis=1),
        'saturation': np.mean(saturation, axis=1),
        'lightness': np.mean(lightness, axis=1),
        'variance': np.var(gray, axis=1)
    }

    # Update raw_profile with horizontally excluded intensity
    raw_profile = profiles['intensity']

    # Extract analysis region
    analysis_region = raw_profile[top_exclude_idx:bottom_exclude_idx]

    # Normalize the profile
    if len(analysis_region) > 0:
        min_val = analysis_region.min()
        max_val = analysis_region.max()
        if max_val > min_val:
            normalized = (analysis_region - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(analysis_region)
    else:
        normalized = np.array([])

    # Create full normalized profile with excluded regions marked
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


def find_brightness_threshold_regions(
        profile: TurbidityProfile,
        threshold: float = TURBIDITY_PARAMS.get('brightness_threshold', 0.5)
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
    """
    Find normalized height(s) where brightness first exceeds `threshold`.

    Returns:
        List of normalized y-positions (0.0 = top, 1.0 = bottom) in analysis region.
    """
    norm = profile.normalized_profile
    excluded = profile.excluded_regions

    # Only consider analysis region
    start_idx = excluded['top_exclude_idx']
    end_idx = excluded['bottom_exclude_idx']
    region_norm = norm[start_idx:end_idx]

    if len(region_norm) == 0:
        return []

    # Map indices back to normalized height in full profile
    full_height = TURBIDITY_PARAMS['analysis_height']
    y_positions = np.linspace(0, 1, full_height)
    region_y = y_positions[start_idx:end_idx]

    # Find first crossing from below
    crossings = []
    crossed = False
    for val, y in zip(region_norm, region_y):
        if not crossed and val > threshold:
            crossings.append(float(y))
            crossed = True  # only first crossing (or remove to get all)

    # Find contiguous segments above threshold
    above = region_norm > threshold
    regions = []
    in_region = False
    start_y = None

    for val, y in zip(above, region_y):
        if val and not in_region:
            start_y = y
            in_region = True
        elif not val and in_region:
            regions.append((float(start_y), float(y)))
            in_region = False

    # Close final region if still open
    if in_region:
        regions.append((float(start_y), float(region_y[-1])))

    return crossings, regions


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="vial_image.jpg")
    parser.add_argument("-o", "--out_dir", type=str, default="output")
    args = parser.parse_args()
    image_path = args.image_path
    out_dir = args.out_dir

    # Add turbidity analysis if image exists
    img = cv2.imread(str(image_path))
    if img is not None:
        profile = compute_turbidity_profile(img)

        # Find where brightness
        crossings, regions = find_brightness_threshold_regions(profile, threshold=TURBIDITY_PARAMS['brightness_threshold'])

        # Save the resized analysis image for region stats
        analysis_img = cv2.resize(img,
                                  (TURBIDITY_PARAMS['analysis_width'],
                                   TURBIDITY_PARAMS['analysis_height']))

        # Extract exclusion fractions
        ex = profile.excluded_regions
        stats = analyze_region_turbidity(
            region_image=analysis_img,
            top_frac=ex['excluded_top_fraction'],
            bottom_frac=ex['excluded_bottom_fraction'],
            left_frac=ex['excluded_left_fraction'],
            right_frac=ex['excluded_right_fraction']
        )

        # Save turbidity plot
        plot_path = save_enhanced_turbidity_plot(
            image_path,
            profile.normalized_profile,
            getattr(profile, "excluded_regions", None),
            out_dir
        )

        # Add turbidity statistics
        results = {
            "mean": float(np.mean(profile.normalized_profile)),
            "std": float(np.std(profile.normalized_profile)),
            "max_gradient": float(np.max(np.abs(np.gradient(profile.normalized_profile))))
            if len(profile.normalized_profile) > 1 else 0.0
        }

        print(f"Turbidity statistics: {results}")
        print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} at normalized height(s): {crossings}")
        print(f"Brightness > {TURBIDITY_PARAMS['brightness_threshold']} in contiguous {len(regions)} regions: {regions}")
        # print(f"Analysis image shape: {analysis_img.shape}")
        # print(f"Exclusion zones: top={ex['excluded_top_fraction']:.2f}, "
        #       f"bottom={ex['excluded_bottom_fraction']:.2f}, "
        #       f"left={ex['excluded_left_fraction']:.2f}, right={ex['excluded_right_fraction']:.2f}")
        # print(f"Region stats: {stats}")