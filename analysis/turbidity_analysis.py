"""
Turbidity profile analysis for phase separation detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from config import TURBIDITY_PARAMS


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
    
    # Convert to grayscale for turbidity analysis
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    
    # Compute row-wise mean intensity
    raw_profile = np.mean(gray, axis=1)
    
    # Determine exclusion regions
    if cap_bottom_y is not None:
        # Scale cap position to analysis dimensions
        cap_bottom_scaled = int((cap_bottom_y / h_original) * TURBIDITY_PARAMS['analysis_height'])
        # Add buffer below cap
        top_exclude_idx = min(cap_bottom_scaled + 10, 
                             TURBIDITY_PARAMS['analysis_height'] // 3)
    else:
        # Use default top exclusion
        top_exclude_idx = int(TURBIDITY_PARAMS['analysis_height'] * 
                             TURBIDITY_PARAMS['top_exclude_fraction'])
    
    # Bottom exclusion for vial bottom artifacts
    bottom_exclude_idx = int(TURBIDITY_PARAMS['analysis_height'] * 
                            (1 - TURBIDITY_PARAMS['bottom_exclude_fraction']))
    
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
        'cap_detected': cap_bottom_y is not None
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


def analyze_region_turbidity(region_image: np.ndarray) -> Dict[str, float]:
    """
    Analyze turbidity statistics for a detection region.
    
    Args:
        region_image: Cropped region image
        
    Returns:
        Dictionary of turbidity statistics
    """
    if region_image.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'max_gradient': 0.0}
    
    # Convert to grayscale
    if len(region_image.shape) == 3:
        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_image
    
    # Compute row-wise mean
    row_means = np.mean(gray, axis=1)
    
    # Normalize
    if row_means.max() > row_means.min():
        normalized = (row_means - row_means.min()) / (row_means.max() - row_means.min())
    else:
        normalized = np.zeros_like(row_means)
    
    # Compute statistics
    stats = {
        'mean': float(np.mean(normalized)),
        'std': float(np.std(normalized)),
        'max_gradient': float(np.max(np.abs(np.gradient(normalized)))) if len(normalized) > 1 else 0.0
    }
    
    return stats