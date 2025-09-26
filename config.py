"""
Configuration module for vial detection pipeline.
Contains all constants, thresholds, and configuration parameters.
"""

from pathlib import Path

# Class IDs from YOLO model
CLASS_IDS = {
    'GEL': 0,
    'STABLE': 1,
    'AIR': 2,
    'CAP': 3
}

# Liquid classes
LIQUID_CLASSES = {CLASS_IDS['STABLE'], CLASS_IDS['GEL']}

# Detection thresholds
DETECTION_THRESHOLDS = {
    'conf_min': 0.20,               # Minimum confidence for detections
    'iou_thr': 0.50,                # IoU threshold for merging boxes
    'gel_area_frac': 0.35,          # Gel area fraction threshold
    'gel_dominance_count': 1,       # Gel box count dominance threshold
}

# Phase separation thresholds
PHASE_SEPARATION_THRESHOLDS = {
    'min_area_frac': 0.002,         # Minimum area fraction for liquid detection
    'gap_thr': 0.03,                # Vertical gap threshold (normalized)
    'span_thr': 0.20,               # Vertical span threshold (normalized)
}

# Turbidity analysis parameters
TURBIDITY_PARAMS = {
    'analysis_width': 100,          # Standard width for turbidity analysis
    'analysis_height': 500,         # Standard height for turbidity analysis
    'top_exclude_fraction': 0.25,   # Default top exclusion fraction
    'bottom_exclude_fraction': 0.05, # Bottom exclusion fraction
    'gradient_threshold_sigma': 2.5, # Sigma multiplier for gradient threshold
    'gradient_threshold_min': 0.10,  # Minimum gradient threshold
    'peak_separation_fraction': 0.1, # Minimum separation between peaks
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'line_thickness': 2,
    'plot_dpi': 120,
    'figure_size': (6, 4),
}

# Default paths
DEFAULT_PATHS = {
    'yolov5_root': Path('yolov5'),
    'output_root': Path('runs/vial2liquid'),
}