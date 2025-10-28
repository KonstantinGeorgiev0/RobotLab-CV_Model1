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

# Liquid detector conf
LIQUID_DETECTOR = {
    'liquid_weights': 'liquid/best_renamed.pt',  # Path to weights file
    'liquid_task': 'detect',  # or 'segment'
    'liquid_img_size': 640,
    'liquid_conf': 0.45,
    'liquid_iou': 0.50,
}

# Vial detector conf
VIAL_DETECTOR = {
    'vial_weights': 'vial/best.pt',
    'vial_imgsz': 640,
    'vial_conf': 0.65,
    'vial_iou': 0.45,
    'vial_pad': 0.05,
    'vial_crop_h': 640,
    'vial_topk': -1
}

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
    'analysis_width': 100,           # Standard width for turbidity analysis
    'analysis_height': 500,          # Standard height for turbidity analysis
    'top_exclude_fraction': 0.25,    # Default top exclusion fraction
    'bottom_exclude_fraction': 0.05, # Bottom exclusion fraction
    'gradient_threshold_sigma': 2.5, # Sigma multiplier for gradient threshold
    'gradient_threshold_min': 0.10,  # Minimum gradient threshold
    'peak_separation_fraction': 0.1, # Minimum separation between peaks
}

# Region exclusion for liquid detection (normalized coordinates)
REGION_EXCLUSION = {
    'top_fraction': 0.20,      # Exclude top %
    'bottom_fraction': 0.10,   # Exclude bottom %
    'enabled': True            # Toggle feature on/off
}

# Line detection parameters
LINE_PARAMS = {
    'min_line_length': 0.75,           # minimum line length for detection
    'merge_threshold': 0.05,           # merge lines that are too close together
    'top_exclusion': 0.30,             # top exclusion fraction
    'bottom_exclusion': 0.15,          # bottom exclusion fraction
    'horizontal_bounds': (0.03, 0.97), # normalized (left, right)
    'search_offset_frac': 0.05,        # vertical search offset around guide line (image fraction)
    'median_kernel': 9,                # median filter kernel size
    'max_step_px': 3,                  # max step between points (pixels)
    # line_hv_detector.py params
    'horiz_kernel_div': 15,            # horizontal kernel size
    'vert_kernel_div': 30,             # vertical kernel size
    'adaptive_block': 15,              # adaptive block
    'adaptive_c': -2,                  # adaptive c
    'min_line_strength': 0.8,          # min line strength
}

# Curve analysis thresholds
CURVE_PARAMS = {
    "gel_variance_thr": 80.0,           # variance threshold for gel detection
    "stable_variance_thr": 50.0,        # variance threshold for stable detection
    "std_dev_thr": 10.0,                # standard deviation threshold
    "roughness_thr": 0.85,              # std of 2nd derivative, optional second guard
    "inter_segment_variance": 40.0,     # inter segment variance
    "min_points": 120,                  # require enough traced points
    "min_line_length": 0.25,            # minimum line length for curve detection
    "vertical_bounds": (0.30, 0.80),    # normalized (top, bottom)
    "horizontal_bounds": (0.05, 0.95),  # normalized (left, right)
    "search_offset_frac": 0.05,         # vertical search offset around guide line (image fraction)
    "median_kernel": 9,                 # median filter kernel size
    "max_step_px": 4,                   # max step between points (pixels)
    "num_segments": 5
}

# Only air classification
ONLY_AIR_CLASSIFICATION = {
    "min_air_height_fraction": 0.6,         # height fraction for air classification
    "min_horizontal_line_len_frac": 0.6,     # length fraction for horizontal line detection
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

LINE_RULES = {
    "cap_level_frac": 0.08,          # fraction of vial height from the top to ignore as cap/neck
    "min_line_len_frac": 0.65,       # length relative to interior width to count as “true” interface
    "two_line_ps_conf": "high",      # sets confidence bump for phase separation
}

REGION_RULES = {
    "full_vial_min_height_frac": 0.85,  # single liquid spans ≥ this of vial height
    "top_half_frac": 0.50,              # boundary for top half
    "bottom_touch_pad_px": 6,           # bottom reach padding
}

DETECTION_FILTERS = {
    "conf_min": 0.25,
    "min_liquid_area_frac": 0.01,   # drop tiny fragments
    "merge_iou": 0.5,               # merge overlapping liquid fragments
}
