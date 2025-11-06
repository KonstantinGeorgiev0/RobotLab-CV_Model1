from typing import Optional

import cv2
import numpy as np
from pathlib import Path
from config import CLASS_IDS, LINE_PARAMS, CURVE_PARAMS
from image_analysis.guided_curve_analysis import GuidedCurveTracer, CurveAnalyzer, CurveVisualizer
from image_analysis.line_hv_detection import LineDetector
from robotlab_utils.bbox_utils import yolo_line_to_xyxy


def create_filtered_detection_visualization(
        crop_path: Path,
        label_path: Path,
        output_path: Path,
        top_fraction: float = 0.0,
        bottom_fraction: float = 0.0,
        show_excluded_regions: bool = True,
        export_scale: float = 1.0,
        supersample: bool = True,
    ) -> Path:
    """
    Create visualization with only non-excluded detections.

    Args:
        crop_path: Path to crop image
        label_path: Path to label file (YOLO format)
        output_path: Where to save visualization
        top_fraction: Top exclusion fraction [0..1]
        bottom_fraction: Bottom exclusion fraction [0..1]
        show_excluded_regions: Draw red overlay on excluded regions
        export_scale: Output size multiplier (e.g., 1.0 = original, 2.0 = 2x larger)
        supersample: If True, draw at 2x then downscale for extra crispness

    Returns:
        Path to saved visualization
    """

    # --- Load image ---
    img = cv2.imread(str(crop_path))
    if img is None:
        return output_path

    H0, W0 = img.shape[:2]

    # draw on larger canvas
    ss = 2.0 if supersample else 1.0
    work_scale = max(1.0, ss * export_scale)

    W = int(round(W0 * work_scale))
    H = int(round(H0 * work_scale))
    work_img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)

    vis_img = work_img.copy()

    # CLASS_IDS handling
    keys = list(CLASS_IDS.keys())
    if keys and isinstance(keys[0], str):
        NAME2ID = CLASS_IDS
        ID2NAME = {v: k for k, v in CLASS_IDS.items()}
    else:
        ID2NAME = CLASS_IDS
        NAME2ID = {v: k for k, v in CLASS_IDS.items()}

    # Colors (BGR)
    COLORS = {
        'GEL':    (0, 165, 255),  # Orange
        'STABLE': (0, 255, 0),    # Green
        'AIR':    (255, 0, 0),    # Blue
        '_OTHER': (128, 128, 128) # Gray
    }

    # Scale-aware styling
    diag = (W * H) ** 0.5
    base = max(0.7, diag / 3000.0)
    box_th = max(2, int(round(2.0 * base)))
    line_th = max(2, int(round(2.0 * base)))
    font_scale = 1.0 * base
    font_th = max(1, int(round(2.0 * base)))
    stroke_th = max(1, int(round(3.0 * base)))  # text outline thickness
    pad = max(4, int(round(8 * base)))          # label padding
    footer_h = max(28, int(round(42 * base)))   # footer bar height
    excl_alpha = 0.18                            # excluded overlay transparency
    label_alpha = 0.85                           # label chip opacity

    # Exclusion boundaries
    top_boundary = int(H * top_fraction)
    bottom_boundary = int(H * (1.0 - bottom_fraction))

    # Draw excluded regions (semi-transparent)
    if show_excluded_regions and (top_fraction > 0 or bottom_fraction > 0):
        overlay = vis_img.copy()
        red = (0, 0, 255)
        if top_fraction > 0:
            cv2.rectangle(overlay, (0, 0), (W, top_boundary), red, -1)
        if bottom_fraction > 0:
            cv2.rectangle(overlay, (0, bottom_boundary), (W, H), red, -1)
        vis_img = cv2.addWeighted(overlay, excl_alpha, vis_img, 1.0 - excl_alpha, 0)

        # boundary lines + labels
        if top_fraction > 0:
            cv2.line(vis_img, (0, top_boundary), (W, top_boundary), red, line_th, cv2.LINE_AA)
            txt = f"Top exclusion: {top_fraction * 100:.0f}%"
            _put_text_with_outline(vis_img, txt, (pad, max(pad, top_boundary - pad)),
                                   font_scale, (255, 255, 255), stroke_th)
        if bottom_fraction > 0:
            cv2.line(vis_img, (0, bottom_boundary), (W, bottom_boundary), red, line_th, cv2.LINE_AA)
            txt = f"Bottom exclusion: {bottom_fraction * 100:.0f}%"
            # place slightly below the line but within image
            y_txt = min(H - pad, bottom_boundary + int(20 * base))
            _put_text_with_outline(vis_img, txt, (pad, y_txt),
                                   font_scale, (255, 255, 255), stroke_th)

    # Draw detections
    kept_count, excluded_count = 0, 0

    if label_path.exists():
        with open(label_path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                parsed = yolo_line_to_xyxy(line, W0, H0)  # parse in original scale
                if not parsed:
                    continue

                cls_id, box_xyxy, conf = parsed
                # scale box to working canvas
                x1, y1, x2, y2 = [int(round(v * work_scale)) for v in box_xyxy]
                center_y = (y1 + y2) / 2.0

                in_excluded = (center_y < top_boundary) or (center_y > bottom_boundary)
                if in_excluded:
                    excluded_count += 1
                    continue

                kept_count += 1

                # class name & color
                class_name = ID2NAME.get(cls_id, f"Class_{cls_id}")
                color = (
                    COLORS.get(class_name)
                    if class_name in COLORS
                    else COLORS['_OTHER']
                )
                # If CLASS_IDS is name->id
                if class_name not in COLORS:
                    # try by comparing ids for known names
                    if cls_id == NAME2ID.get('GEL'):
                        color = COLORS['GEL']
                    elif cls_id == NAME2ID.get('STABLE'):
                        color = COLORS['STABLE']
                    elif cls_id == NAME2ID.get('AIR'):
                        color = COLORS['AIR']

                # box (anti-aliased)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, box_th, cv2.LINE_AA)

                # label text
                label = f"{class_name} {conf:.2f}"

                # measure text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)
                tw += 2 * pad
                th += 2 * pad

                # prefer above, fallback below if needed
                label_x1 = x1
                label_y2 = y1 - 2
                label_y1 = label_y2 - th

                if label_y1 < 0:  # move below box
                    label_y1 = y2 + 2
                    label_y2 = label_y1 + th

                # label chip (semi-transparent)
                lx1 = max(0, label_x1)
                lx2 = min(W, label_x1 + tw)
                ly1 = max(0, label_y1)
                ly2 = min(H, label_y2)

                if lx2 > lx1 and ly2 > ly1:
                    chip = vis_img[ly1:ly2, lx1:lx2].copy()
                    chip_overlay = chip.copy()
                    cv2.rectangle(chip_overlay, (0, 0), (lx2 - lx1, ly2 - ly1), color, -1)
                    vis_img[ly1:ly2, lx1:lx2] = cv2.addWeighted(
                        chip_overlay, label_alpha, chip, 1.0 - label_alpha, 0
                    )

                    # white text with black outline for contrast
                    tx = lx1 + pad
                    ty = ly1 + th - pad
                    _put_text_with_outline(vis_img, label, (tx, ty), font_scale,
                                           (255, 255, 255), stroke_th, font_th)

    # Footer summary bar
    footer = np.zeros((footer_h, W, 3), dtype=np.uint8)
    footer[:] = (25, 25, 25)
    summary = f"Kept: {kept_count} | Excluded: {excluded_count}"
    _put_text_with_outline(footer, summary, (pad, int(footer_h * 0.7)),
                           font_scale * 1.1, (255, 255, 255), stroke_th, font_th)
    vis_img = np.vstack([vis_img, footer])

    # Final export size
    target_W = int(round(W0 * export_scale))
    scale_factor = target_W / vis_img.shape[1]
    target_H = int(round(vis_img.shape[0] * scale_factor))
    out = cv2.resize(
        vis_img,
        (target_W, target_H),
        interpolation=cv2.INTER_AREA if supersample else cv2.INTER_CUBIC
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return output_path


def _put_text_with_outline(
    img,
    text: str,
    org: tuple,
    font_scale: float,
    color: tuple,
    outline_th: int = 2,
    font_th: int = 1,
):
    """
    Draws high-contrast text: black outline + filled color text.
    """
    x, y = org
    # outline (black)
    for dx in (-outline_th, 0, outline_th):
        for dy in (-outline_th, 0, outline_th):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (x + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        max(1, outline_th), cv2.LINE_AA)
    # fill
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                max(1, font_th), cv2.LINE_AA)


def create_line_visualization(
    image_path: Path,
    output_path: Path,
    top_exclusion: float = 0.28,
    bottom_exclusion: float = 0.15
) -> Path:
    """
    Create and save visualization of horizontal/vertical lines.

    Args:
        image_path: Path to input image (e.g., crop)
        output_path: Where to save the visualization PNG
        top_exclusion: Fraction of top to exclude (0-1)
        bottom_exclusion: Fraction of bottom to exclude (0-1)

    Returns:
        Path to saved visualization
    """
    # Initialize with params from config
    detector = LineDetector(
        horiz_kernel_div=LINE_PARAMS.get("horiz_kernel_div", 15),
        vert_kernel_div=LINE_PARAMS.get("vert_kernel_div", 30),
        adaptive_block=LINE_PARAMS.get("adaptive_block", 15),
        adaptive_c=LINE_PARAMS.get("adaptive_c", -2),
        min_line_length=LINE_PARAMS.get("min_line_length", 0.3),
        min_line_strength=LINE_PARAMS.get("min_line_strength", 0.1),
        merge_threshold_horizontal=LINE_PARAMS.get("merge_threshold_horizontal", 0.02),
        merge_threshold_vertical=LINE_PARAMS.get("merge_threshold_vertical", 0.02),
    )
    result = detector.detect(image_path, top_exclusion=top_exclusion, bottom_exclusion=bottom_exclusion)
    detector.visualize(image_path, output_path, result, top_exclusion, bottom_exclusion)
    return output_path


def create_curve_visualization(
    image_path: Path,
    output_path: Path,
    top: float = CURVE_PARAMS["vertical_bounds"][0],
    bottom: float = CURVE_PARAMS["vertical_bounds"][1],
    left: float = CURVE_PARAMS["horizontal_bounds"][0],
    right: float = CURVE_PARAMS["horizontal_bounds"][1],
    search_offset: float = CURVE_PARAMS["search_offset_frac"],
    median_k: int = CURVE_PARAMS["median_kernel"],
    max_step: int = CURVE_PARAMS["max_step_px"],
    guide_y: Optional[float] = None,
    window_size: int = 20,
    anomaly_threshold: float = 3.0,
    num_segments: int = 3
) -> Path:
    """
    Create and save visualizations of guided curve tracer, including analysis from CurveAnalyzer.

    Args:
        image_path: Path to input image (e.g., crop)
        output_path: Base path for saving visualizations (will append suffixes like '_guided_curve.png')
        top: Top boundary fraction (0-1)
        bottom: Bottom boundary fraction (0-1)
        left: Left boundary fraction (0-1)
        right: Right boundary fraction (0-1)
        search_offset: Vertical search offset fraction (of image height)
        median_k: Median filter kernel size
        max_step: Max step between points (pixels)
        guide_y: Optional guide y-position (0-1)
        window_size: Window size for local variance analysis
        anomaly_threshold: Sigma threshold for anomaly detection
        num_segments: Number of segments for curve segmentation

    Returns:
        Base Path for the visualizations (individual files saved with suffixes)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    tracer = GuidedCurveTracer(
        vertical_bounds=(top, bottom),
        horizontal_bounds=(left, right),
        search_offset_frac=search_offset,
        median_kernel=median_k,
        max_step_px=max_step
    )
    xs, ys, metadata = tracer.trace_curve(img, image_path, guide_y=guide_y)

    # Save basic guided curve visualization
    guided_curve_path = output_path.with_stem(output_path.stem + '_guided_curve')
    tracer.visualize(img, xs, ys, metadata, guided_curve_path)

    # Run analysis
    analyzer = CurveAnalyzer()
    baseline_y = metadata.get('guide_y_px')  # Use guide as baseline
    curve_stats = analyzer.compute_comprehensive_statistics(
        xs=xs,
        ys=ys,
        baseline_y=baseline_y,
        window_size=window_size
    )
    anomalies = analyzer.detect_anomalies(
        xs=xs,
        ys=ys,
        threshold_sigma=anomaly_threshold
    )
    segments = analyzer.segment_curve(
        xs=xs,
        ys=ys,
        num_segments=num_segments
    )

    # Save additional visualizations
    visualizer = CurveVisualizer()

    overlay_path = output_path.with_stem(output_path.stem + '_overlay')
    visualizer.create_image_overlay(
        img=img,
        xs=xs,
        ys=ys,
        stats=curve_stats,
        curve_metadata=metadata,
        anomalies=anomalies,
        output_path=overlay_path
    )

    analytics_path = output_path.with_stem(output_path.stem + '_analytics')
    visualizer.create_analytics_plot(
        img=img,
        xs=xs,
        ys=ys,
        stats=curve_stats,
        curve_metadata=metadata,
        anomalies=anomalies,
        segments=segments,
        output_path=analytics_path
    )

    return output_path