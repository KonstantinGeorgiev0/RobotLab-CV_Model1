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
    'air_brightness_thr': 100.0,    # Mean brightness below this → likely AIR
    'empty_top_frac': 0.35,         # If top y1 > height * this, consider empty top
    'large_liquid_bottom_frac': 0.75,  # If single liquid extends > this frac of height, check for misclassification
}

# Phase separation thresholds
PHASE_SEPARATION_THRESHOLDS = {
    'min_area_frac': 0.002,         # Minimum area fraction for liquid detection
    'gap_thr': 0.03,                # Vertical gap threshold (normalized)
    'span_thr': 0.20,               # Vertical span threshold (normalized)
}

# Turbidity analysis parameters
TURBIDITY_PARAMS = {
    'analysis_width': 120,           # Standard width for turbidity analysis
    'analysis_height': 600,          # Standard height for turbidity analysis
    'top_exclude_fraction': 0.30,    # Default top exclusion fraction
    'bottom_exclude_fraction': 0.10, # Bottom exclusion fraction
    'right_exclude_fraction': 0.25,  # Right exclusion fraction
    'left_exclude_fraction': 0.25,   # Left exclusion fraction
    'gradient_threshold_sigma': 2.5, # Sigma multiplier for gradient threshold
    'gradient_threshold_min': 0.15,  # Minimum gradient threshold
    'peak_separation_fraction': 0.05, # Minimum separation between peaks
    'brightness_threshold': 0.5,     # Detect where brightness goes above this
    'centerline_width': 15,          # Width of centerline (pixels)
    "air_max": 0.15,                 # air/empty
    "translucent_max": 0.25,         # translucent liquid
    "min_segment_height_frac": 0.01,  # ignore tiny slices
    "liquid_liquid_contrast": 0.10,   # min |Δμ| for liquid–liquid separation

    "min_region_size": 15,
    "gradient_threshold": 0.04,
    "similarity_threshold": 0.15,

    "min_liquid_interfaces": 1,
    "min_vertical_span": 0.15,

    "min_intensity_change": 0.15,    # min change of brightness intensity
    "min_span_fraction": 0.05,       # min vertical span (fraction of analysis height) to ignore noise
    "max_span_fraction": 0.40,       # max vertical span (fraction of analysis height) to still be considered sudden
    "smoothing_sigma": 1.0,          # Gaussian smoothing sigma for brightness profile before change detection
    "gradient_epsilon": 0.05         # Gradient epsilon for change detection
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
    'merge_threshold_horizontal': 0.038,    # merge horiz lines that are too close together
    'merge_threshold_vertical': 0.30,  # merge vert lines
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
    "gel_variance_thr": 75.0,           # variance threshold for gel detection
    "stable_variance_thr": 55.0,        # variance threshold for stable detection
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
    "cap_level_frac": 0.25,          # fraction of vial height from the top to ignore as cap/neck
    "min_line_len_frac": 0.65,       # length relative to interior width to count as “true” interface
    "two_line_ps_conf": "high",      # sets confidence bump for phase separation
}

REGION_RULES = {
    "full_vial_min_height_frac": 0.85,  # single liquid spans ≥ this of vial height
    "top_half_frac": 0.50,              # boundary for top half
    "bottom_touch_pad_px": 6,           # bottom reach padding
    "air_top_touch_frac": 0.25,         # touches top
    "air_deep_span_frac": 0.75,         # spans deep down the vial
    "min_headspace_frac": 0.15,         # minimum headspace fraction required
    "vertical_gap_frac": 0.075,         # vertical gap between AIR bottom and LIQUID top
    "vertical_gap_frac_with_bottom": 0.10 # vertical gap between LIQUID bottom and image bottom
}

DETECTION_FILTERS = {
    "conf_min": 0.25,
    "min_liquid_area_frac": 0.05,    # drop small fragments
    "merge_iou": 0.5,               # merge overlapping liquid fragments
}
