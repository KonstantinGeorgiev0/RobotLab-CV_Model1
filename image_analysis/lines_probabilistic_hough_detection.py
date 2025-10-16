#!/usr/bin/env python3
# lines_probabilistic_hough_detection.py (headless)
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

from robotlab_utils.image_utils import extract_edges_for_curve_detection


OUTDIR = Path("curved_line_detection_results")
OUTDIR.mkdir(parents=True, exist_ok=True)


def waviness_from_segments(segments, w_norm):
    """
    'wiggle' metric: fit a straight line y=ax+b over
    all segment midpoints, compute RMS deviation normalized by width.
    """
    if not segments:
        return 0.0
    mids = []
    for (x1, y1, x2, y2) in segments:
        mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    mids = np.asarray(mids, dtype=np.float32)
    X = np.hstack([mids[:, 0:1], np.ones((mids.shape[0], 1), np.float32)])  # [x, 1]
    y = mids[:, 1]
    # least squares y = a*x + b
    sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = (sol[0] * mids[:, 0] + sol[1])
    rms = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    wiggle = rms / max(1.0, w_norm)
    return wiggle


def filter_near_horizontal_segments(segments, max_angle_deg=10.0):
    """
    Filter segments to keep only near-horizontal ones.

    Args:
        segments: List of (x1, y1, x2, y2) tuples
        max_angle_deg: Maximum absolute angle from horizontal in degrees

    Returns:
        List of filtered segments
    """
    filtered = []
    for (x1, y1, x2, y2) in segments:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= max_angle_deg:
            filtered.append((x1, y1, x2, y2))
    return filtered


def detect_hough_lines(edges,
                       min_line_length_frac=0.1,
                       max_line_gap=5,
                       hough_threshold=30,
                       filter_horizontal=True,
                       max_angle=10.0):
    """
    Detect line segments using Probabilistic Hough Transform.

    Args:
        edges: Binary edge image
        min_line_length_frac: Minimum line length as fraction of image width
        max_line_gap: Maximum gap between points to be considered same line
        hough_threshold: Accumulator threshold for Hough transform
        filter_horizontal: Whether to filter for near-horizontal lines
        max_angle: Maximum angle from horizontal (if filter_horizontal=True)

    Returns:
        Tuple of (all_segments, horizontal_segments, wiggle_metric)
    """
    H, W = edges.shape[:2]

    # Calculate minimum line length in pixels
    min_line_length = max(30, int(min_line_length_frac * W))

    # Run Probabilistic Hough Transform
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Convert to list of tuples
    all_segments = []
    if raw is not None:
        for s in raw:
            x1, y1, x2, y2 = map(int, s[0])
            all_segments.append((x1, y1, x2, y2))

    # Filter for horizontal segments if requested
    if filter_horizontal:
        horizontal_segments = filter_near_horizontal_segments(all_segments, max_angle)
    else:
        horizontal_segments = all_segments

    # Calculate wiggle metric
    wiggle = waviness_from_segments(horizontal_segments, W)

    return all_segments, horizontal_segments, wiggle


def apply_region_mask(img, top_margin_frac=0.0, bottom_margin_frac=0.0):
    """
    Apply mask to exclude top and bottom regions of image.

    Args:
        img: Input image
        top_margin_frac: Fraction of height to exclude from top (0-1)
        bottom_margin_frac: Fraction of height to exclude from bottom (0-1)

    Returns:
        Masked image
    """
    if top_margin_frac == 0.0 and bottom_margin_frac == 0.0:
        return img

    H, W = img.shape[:2]
    top_pixels = int(H * top_margin_frac)
    bottom_pixels = int(H * bottom_margin_frac)

    # Create mask for valid image region
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top_pixels:H - bottom_pixels, :] = 255

    # Apply mask to image
    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


def visualize_hough_lines(img, segments, wiggle, output_path):
    """
    Create visualization with detected line segments.

    Args:
        img: Input image
        segments: List of (x1, y1, x2, y2) line segments
        wiggle: Wiggle metric value
        output_path: Path to save visualization
    """
    overlay = img.copy()

    # Draw line segments
    for (x1, y1, x2, y2) in segments:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    # Add text annotation
    text = f"horiz_lines={len(segments)}  wiggle={wiggle:.3f}"
    cv2.putText(overlay, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Save
    cv2.imwrite(str(output_path), overlay)


def main():
    parser = argparse.ArgumentParser(
        description="Detect horizontal lines using Probabilistic Hough Transform"
    )
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--outdir", default="curved_line_detection_results",
                        help="output directory for results")
    parser.add_argument("--top-margin", type=float, default=0.1,
                        help="fraction of image height to exclude from top (default: 0.1)")
    parser.add_argument("--bottom-margin", type=float, default=0.1,
                        help="fraction of image height to exclude from bottom (default: 0.1)")
    parser.add_argument("--canny", nargs=2, type=int, default=[15, 70],
                        help="Canny thresholds (low high)")
    parser.add_argument("--denoise", choices=['bilateral', 'epf', 'nlmeans', 'gaussian', 'none'],
                        default='bilateral', help="Denoising method")
    parser.add_argument("--min-line-length", type=float, default=0.1,
                        help="Minimum line length as fraction of image width")
    parser.add_argument("--max-line-gap", type=int, default=5,
                        help="Maximum gap between line segments")
    parser.add_argument("--hough-threshold", type=int, default=30,
                        help="Hough transform accumulator threshold")
    parser.add_argument("--max-angle", type=float, default=10.0,
                        help="Maximum angle from horizontal (degrees)")
    args = parser.parse_args()

    # Setup output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read image
    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read: {img_path}")

    # Apply region mask to exclude top and bottom margins
    masked_img = apply_region_mask(img, args.top_margin, args.bottom_margin)

    # Extract edges using centralized preprocessing
    edges = extract_edges_for_curve_detection(
        masked_img,
        canny_low=args.canny[0],
        canny_high=args.canny[1],
        denoise_method=args.denoise,
        morphology_close=True,
        morphology_dilate=True
    )

    # Detect lines using Hough transform
    all_segments, horiz_segments, wiggle = detect_hough_lines(
        edges,
        min_line_length_frac=args.min_line_length,
        max_line_gap=args.max_line_gap,
        hough_threshold=args.hough_threshold,
        filter_horizontal=True,
        max_angle=args.max_angle
    )

    # Create and save visualizations
    stem = img_path.stem

    # Save edge image
    edges_path = outdir / f"{stem}_curved_edges.png"
    cv2.imwrite(str(edges_path), edges)

    # Save overlay with detected lines
    overlay_path = outdir / f"{stem}_curved_overlay.png"
    visualize_hough_lines(img, horiz_segments, wiggle, overlay_path)

    # Print results
    print(f"\nResults for {img_path.name}:")
    print(f"  Total segments detected: {len(all_segments)}")
    print(f"  Near-horizontal segments: {len(horiz_segments)}")
    print(f"  Wiggle metric: {wiggle:.3f}")
    print(f"\nSaved outputs:")
    print(f"  Edges: {edges_path}")
    print(f"  Overlay: {overlay_path}")


if __name__ == "__main__":
    main()