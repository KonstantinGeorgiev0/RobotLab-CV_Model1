import argparse
import sys
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.turbidity.turbidity_metrics import analyze_region_turbidity
from analysis.turbidity.turbidity_profiles import compute_turbidity_profile, compute_variance_between_changes, \
    compute_centerline_turbidity_profile
from analysis.turbidity.turbidity_segmentation import find_brightness_threshold_regions, segment_brightness_regions, \
    detect_sudden_brightness_changes
from analysis.turbidity.turbidity_separation import label_segments, detect_separation_types, \
    detect_phase_separation_from_separations

import cv2
import numpy as np

from config import TURBIDITY_PARAMS
from visualization.turbidity_viz import save_turbidity_plot, save_turbidity_plot_analysis_only


def extract_turbidity_features(img: np.ndarray, params) -> Dict[str, Any]:
    """
    Extract all turbidity-related features from a single vial image
    """

    # compute turbidity profiles
    full_profile = compute_turbidity_profile(img)
    centerline_width = getattr(params, "centerline_width", TURBIDITY_PARAMS["centerline_width"])
    center_profile = compute_centerline_turbidity_profile(
        img,
        centerline_width=centerline_width,
    )

    # global turbidity stats using the full exclusion fractions
    ex_full = full_profile.excluded_regions
    top_frac = ex_full.get("excluded_top_fraction", 0.0)
    bottom_frac = ex_full.get("excluded_bottom_fraction", 0.0)
    left_frac = ex_full.get("excluded_left_fraction", 0.0)
    right_frac = ex_full.get("excluded_right_fraction", 0.0)

    global_stats = analyze_region_turbidity(
        img,
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        left_frac=left_frac,
        right_frac=right_frac,
    )

    # brightness threshold analysis for full and centerline profiles
    brightness_threshold = getattr(
        params,
        "brightness_threshold",
        TURBIDITY_PARAMS.get("brightness_threshold", 0.5),
    )

    full_crossings, full_bright_regions = find_brightness_threshold_regions(
        full_profile,
        threshold=brightness_threshold,
    )

    center_crossings, center_bright_regions = find_brightness_threshold_regions(
        center_profile,
        threshold=brightness_threshold,
    )

    # sudden brightness changes along centerline
    sudden_changes = detect_sudden_brightness_changes(
        center_profile,
        min_intensity_change=TURBIDITY_PARAMS["min_intensity_change"],
        min_span_fraction=TURBIDITY_PARAMS["min_span_fraction"],
        max_span_fraction=TURBIDITY_PARAMS["max_span_fraction"],
        smoothing_sigma=TURBIDITY_PARAMS["smoothing_sigma"],
        gradient_epsilon=TURBIDITY_PARAMS["gradient_epsilon"],
    )

    # variance and segment stats between sudden changes
    variance_stats = compute_variance_between_changes(
        center_profile,
        sudden_changes,
    )
    segments = variance_stats.get("segments", [])

    brightness_segments = segment_brightness_regions(
        center_profile,
        similarity_threshold=TURBIDITY_PARAMS["similarity_threshold"],
        min_region_size=TURBIDITY_PARAMS["min_region_size"],
        smoothing_sigma=TURBIDITY_PARAMS["smoothing_sigma"],
        gradient_threshold=TURBIDITY_PARAMS["gradient_threshold"],
    )

    # label segments with phases and detect separation events
    analysis_height = center_profile.excluded_regions.get("analysis_height", len(center_profile.raw_profile))

    labeled_segments = label_segments(
        brightness_segments,
        analysis_height=analysis_height,
    )

    separation_events = detect_separation_types(
        labeled_segments,
    )

    # phase separation decision based on separation events
    phase_sep, phase_sep_info = detect_phase_separation_from_separations(
        separation_events,
        min_liquid_interfaces=TURBIDITY_PARAMS["min_liquid_interfaces"],
        min_vertical_span=TURBIDITY_PARAMS["min_vertical_span"],
    )

    # features
    features = {
        # profiles
        "turbidity_profile": full_profile,
        "centerline_profile": center_profile,

        # global stats
        "global_stats": global_stats,

        # threshold-based features
        "full_threshold_crossings": full_crossings,
        "full_bright_regions": full_bright_regions,
        "center_threshold_crossings": center_crossings,
        "center_bright_regions": center_bright_regions,

        # sudden changes and variance
        "sudden_changes": sudden_changes,
        "variance_stats": variance_stats,
        "segments": segments,

        # phase labeling and separation
        "labeled_segments": labeled_segments,
        "separation_events": separation_events,
        "dict_separation_events": [event.to_dict() for event in separation_events],
        "phase_separated": phase_sep,
        "phase_separation_info": phase_sep_info,
    }

    return features

def features_to_json_dict(
    image_path: Path,
    features: Dict[str, Any],
    state_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert turbidity features into a JSON dict
    """
    global_stats = features.get("global_stats", {})
    variance_stats = features.get("variance_stats", {})
    phase_separated = bool(features.get("phase_separated", False))
    phase_sep_info = features.get("phase_separation_info", {})
    sudden_brightness_changes = features.get("sudden_changes", [])

    # convert ifaces to json
    interfaces_json, air_iface_json = (
        [interface.to_dict() for interface in phase_sep_info.get("interfaces", [])],
        [air_iface.to_dict() for air_iface in phase_sep_info.get("air_interface", [])]
    )
    phase_sep_info["interfaces"] = interfaces_json
    phase_sep_info["air_interface"] = air_iface_json

    return {
        "image": str(image_path),
        "state": state_label,
        "global_stats": {
            "mean": float(global_stats.get("mean", 0.0)),
            "std": float(global_stats.get("std", 0.0)),
            "variance": float(global_stats.get("variance", 0.0)),
            "max_gradient": float(global_stats.get("max_gradient", 0.0)),
            "median": float(global_stats.get("median", 0.0)),
            "min": float(global_stats.get("min", 0.0)),
            "max": float(global_stats.get("max", 0.0)),
            "dynamic_range": float(global_stats.get("dynamic_range", 0.0)),
            "gradient_mean": float(global_stats.get("gradient_mean", 0.0)),
            "gradient_std": float(global_stats.get("gradient_std", 0.0)),
            "length": float(global_stats.get("length", 0.0)),
        },
        "overall_variance": float(variance_stats.get("overall_variance", 0.0)),
        "phase_separated": phase_separated,
        "phase_separation_info": phase_sep_info,
        "sudden_brightness_changes": sudden_brightness_changes,
    }


def print_turbidity_report(features: Dict[str, Any], params) -> None:
    """
    Print a summary of turbidity features.
    """

    full_profile = features["turbidity_profile"]
    center_profile = features["centerline_profile"]
    global_stats = features["global_stats"]
    full_crossings = features["full_threshold_crossings"]
    full_bright_regions = features["full_bright_regions"]
    center_crossings = features["center_threshold_crossings"]
    center_bright_regions = features["center_bright_regions"]
    sudden_changes = features["sudden_changes"]
    variance_stats = features["variance_stats"]
    segments = features["segments"]
    labeled_segments = features["labeled_segments"]
    separation_events = features["separation_events"]
    phase_separated = features["phase_separated"]
    phase_sep_info = features["phase_separation_info"]

    ex_full = full_profile.excluded_regions
    ex_center = center_profile.excluded_regions

    print("\n=== turbidity statistics ===")
    for key, val in global_stats.items():
        print(f"{key:15s}: {val}")

    print("\n=== centerline turbidity analysis ===")
    print(f"centerline position: x={ex_center.get('centerline_x', 'n/a')}")
    print(f"centerline width: {ex_center.get('centerline_width', 'n/a')} pixels")

    print(f"\n=== brightness analysis ===")
    threshold = getattr(params, "brightness_threshold", TURBIDITY_PARAMS.get("brightness_threshold", 0.5))
    print(f"First instance brightness > {threshold} at normalized height: {center_crossings}")
    print(
        f"brightness > {threshold} "
        f"in contiguous {len(center_bright_regions)} regions"
    )
    for region in center_bright_regions:
        print(f"\nRegion start={region['start_absolute']:.0f}, end={region['end_absolute']:.0f}")

    print(f"\n=== sudden brightness changes (centerline) ===")
    print(f"number of sudden changes: {len(sudden_changes)}")
    for i, ev in enumerate(sudden_changes):
        print(
            f"  change {i}: "
            f"{ev['direction']} | "
            f"start_norm={ev['start_norm']:.3f}, end_norm={ev['end_norm']:.3f}, "
            f"start_absolute={ev['start_absolute']:.0f}, end_absolute={ev['end_absolute']:.0f},"
            f"ΔI={ev['intensity_change']:.3f}, "
            f"span={ev['span_pixels']} px ({ev['span_fraction']:.3f} of analysis height)"
        )

    print("\n=== variance analysis (centerline, between sudden changes) ===")
    print(f"overall variance in analysis region: {variance_stats['overall_variance']:.4f}")
    print("segments between changes:")
    for seg in segments:
        print(
            f"  segment {seg['segment_id']}: "
            f"{seg['start_normalized']:.3f} → {seg['end_normalized']:.3f}, "
            f"len={seg['height_pixels']} px, "
            f"mean={seg['mean_brightness']:.3f}, "
            f"var={seg['variance']:.4f}"
        )

    print("\n=== separation types from brightness profile ===")
    full_h = TURBIDITY_PARAMS['analysis_height']
    for event in separation_events:
        pixel_y = int(event.boundary_norm * full_h)
        print(
            f"  {event.type} at y={event.boundary_norm:.3f} (pixel {pixel_y}) "
            f"(Δμ={event.delta_brightness:.3f}, {event.top_phase} → {event.bottom_phase})"
        )

    print("\n=== phase separation decision ===")
    print(f"phase separated: {phase_separated}")
    # print(f"phase separation info: {phase_sep_info}")

def parse_args() -> argparse.Namespace:
    """
    parse command line arguments for turbidity analysis.
    """
    parser = argparse.ArgumentParser(description="turbidity analysis for vial images")

    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="path to input vial image (bgr)",
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default="output",
        help="directory to save plots and results",
    )
    parser.add_argument(
        "--centerline_width",
        type=int,
        default=TURBIDITY_PARAMS["centerline_width"],
        help="width of centerline strip in pixels",
    )
    parser.add_argument(
        "--brightness_threshold",
        type=float,
        default=TURBIDITY_PARAMS.get("brightness_threshold", 0.5),
        help="brightness threshold for high-brightness region detection",
    )
    parser.add_argument(
        "--gradient_threshold",
        type=float,
        default=TURBIDITY_PARAMS.get("gradient_threshold", 0.05),
        help="minimum gradient change to consider a sudden brightness change",
    )
    parser.add_argument(
        "--min_liquid_interfaces",
        type=int,
        default=TURBIDITY_PARAMS.get("min_liquid_interfaces", 1),
        help="minimum number of liquid-liquid interfaces for phase separation",
    )
    parser.add_argument(
        "--min_vertical_span",
        type=float,
        default=TURBIDITY_PARAMS.get("min_vertical_span"),
        help="minimum vertical span of liquid interfaces (normalized) for phase separation",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save turbidity metrics as JSON next to the plot",
    )
    parser.add_argument(
        "--state-label",
        type=str,
        default=None,
        help="optional ground-truth state label (stable/gel/phase_separated/only_air)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"failed to load image: {image_path}")

    # extract features
    features = extract_turbidity_features(img, args)

    # print report
    print_turbidity_report(features, args)

    file_dir = Path(args.out_dir) / image_path.stem
    file_dir.mkdir(parents=True, exist_ok=True)

    # plot turbidity profile
    center_profile = features["centerline_profile"]
    plot_path = save_turbidity_plot(
        path=image_path,
        v_norm=center_profile.normalized_profile,
        turbidity_profile=center_profile,
        excluded_info=center_profile.excluded_regions,
        out_dir=file_dir,
        change_events=features.get("sudden_changes"),
        suffix=".turbidity.png",
        use_normalized_height=True,
    )
    print(f"\nsaved turbidity plot to: {plot_path}")

    # plot turbidity of analysis region only
    analysis_region_plot = save_turbidity_plot_analysis_only(
        path=image_path,
        v_norm=center_profile.normalized_profile,
        excluded_info=center_profile.excluded_regions,
        out_dir=file_dir,
    )
    print(f"\nsaved turbidity plot of analysis region to: {analysis_region_plot}")

    # JSON export
    if getattr(args, "save_json", False):
        json_obj = features_to_json_dict(
            image_path=image_path,
            features=features,
            state_label=getattr(args, "state_label", None),
        )
        json_path = file_dir / f"{image_path.stem}.turbidity.json"
        with open(json_path, "w") as f:
            json.dump(json_obj, f, indent=2)
        print(f"saved turbidity metrics to: {json_path}")


if __name__ == "__main__":
    main()
