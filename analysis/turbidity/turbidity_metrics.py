import cv2
import numpy as np
from typing import Dict, List, Any

from analysis.turbidity.turbidity_profiles import TurbidityProfile


def analyze_region_turbidity(
    region_image: np.ndarray,
    top_frac: float = 0.0,
    bottom_frac: float = 0.0,
    left_frac: float = 0.0,
    right_frac: float = 0.0
) -> Dict[str, float]:
    """
    Analyze turbidity in a region with configurable exclusions
    """
    zero_stats = {
        'mean': 0.0,
        'std': 0.0,
        'max_gradient': 0.0,
        'variance': 0.0,
        'median': 0.0,
        'min': 0.0,
        'max': 0.0,
        'dynamic_range': 0.0,
        'gradient_mean': 0.0,
        'gradient_std': 0.0,
        'length': 0.0,
    }

    if region_image.size == 0:
        return zero_stats

    h, w = region_image.shape[:2]

    # apply exclusions
    top_idx = int(h * top_frac)
    bottom_idx = int(h * (1 - bottom_frac))
    left_idx = int(w * left_frac)
    right_idx = int(w * (1 - right_frac))

    # guard against invalid indices
    if bottom_idx <= top_idx or right_idx <= left_idx:
        return zero_stats

    # crop with exclusions
    cropped = region_image[top_idx:bottom_idx, left_idx:right_idx]
    if cropped.size == 0:
        return zero_stats

    # convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

    # row-wise means
    row_means = np.mean(gray, axis=1)
    if row_means.size == 0:
        return zero_stats

    # normalize
    min_val = float(row_means.min())
    max_val = float(row_means.max())
    if max_val > min_val:
        normalized = (row_means - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(row_means, dtype=float)

    # gradient-based measures
    if normalized.size > 1:
        grad = np.gradient(normalized)
        abs_grad = np.abs(grad)
        max_grad = float(abs_grad.max())
        grad_mean = float(abs_grad.mean())
        grad_std = float(abs_grad.std())
    else:
        max_grad = 0.0
        grad_mean = 0.0
        grad_std = 0.0

    # stats
    turbidity_stats = {
        'mean': float(np.mean(normalized)),
        'std': float(np.std(normalized)),
        'max_gradient': max_grad,
        'variance': float(np.var(normalized)),
        'median': float(np.median(normalized)),
        'min': float(normalized.min()) if normalized.size > 0 else 0.0,
        'max': float(normalized.max()) if normalized.size > 0 else 0.0,
        'dynamic_range': float(normalized.max() - normalized.min()) if normalized.size > 0 else 0.0,
        'gradient_mean': grad_mean,
        'gradient_std': grad_std,
        'length': float(normalized.size),
    }
    return turbidity_stats
