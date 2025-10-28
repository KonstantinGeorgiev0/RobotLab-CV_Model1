"""
Guided curve tracing for liquid-air interface detection.
Uses horizontal line detection to guide the search region.
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import Optional, Tuple

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_analysis.line_hv_detection import LineDetector
from robotlab_utils.image_utils import extract_edges_for_curve_detection
from config import CURVE_PARAMS


class GuidedCurveTracer:
    """Traces curved liquid-air interface using horizontal line detection as guide."""

    def __init__(self,
                 vertical_bounds: Tuple[float, float] = (0.20, 0.80),
                 horizontal_bounds: Tuple[float, float] = (0.05, 0.95),
                 search_offset_px: int = 30,
                 median_kernel: int = 9,
                 max_step_px: int = 4):
        """
        Initialize guided curve tracer.

        Args:
            vertical_bounds: (top_frac, bottom_frac) - fraction of image height
            horizontal_bounds: (left_frac, right_frac) - fraction of image width
            search_offset_px: Vertical offset around detected horizontal line to search
            median_kernel: Kernel size for median smoothing
            max_step_px: Maximum allowed step between adjacent points
        """
        self.vertical_bounds = vertical_bounds
        self.horizontal_bounds = horizontal_bounds
        self.search_offset_px = search_offset_px
        self.median_kernel = median_kernel
        self.max_step_px = max_step_px
        self.line_detector = LineDetector(min_line_length=CURVE_PARAMS.get("min_line_length", 0.25))

    def trace_curve(self,
                    img_bgr: np.ndarray,
                    img_path: Path,
                    guide_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Trace curved liquid-air interface.

        Args:
            img_bgr: Input BGR image
            guide_y: Optional normalized y-position (0-1) to guide search.
                    If None, will use horizontal line detection.

        Returns:
            Tuple of (x_coords, y_coords, metadata)
            :param img_path:
        """
        H, W = img_bgr.shape[:2]

        # Detect horizontal line if guide not provided
        if guide_y is None:
            guide_y = self._detect_guide_line(img_bgr, img_path)

        # Convert guide to pixel coordinates
        guide_y_px = int(guide_y * H)

        # Define search region
        x_min_px = int(self.horizontal_bounds[0] * W)
        x_max_px = int(self.horizontal_bounds[1] * W)

        y_min_search = max(0, guide_y_px - self.search_offset_px)
        y_max_search = min(H - 1, guide_y_px + self.search_offset_px)

        # Extract edges optimized for curved interface
        edges = extract_edges_for_curve_detection(img_bgr)

        # Trace curve column by column
        xs, ys = self._trace_columns(
            edges,
            x_min_px, x_max_px,
            y_min_search, y_max_search,
            guide_y_px
        )

        # Smooth and denoise
        ys_smooth = self._smooth_curve(ys)
        y_variance = np.var(ys_smooth)
        deviations_from_guide = ys_smooth - guide_y_px
        variance_from_baseline = np.var(deviations_from_guide)
        std_dev_from_baseline = np.std(deviations_from_guide)
        rms_deviation = np.sqrt(np.mean(deviations_from_guide ** 2))

        # Prepare metadata
        metadata = {
            'y_variance': float(y_variance),
            'std_dev_from_baseline': float(std_dev_from_baseline),
            'variance_from_baseline': float(variance_from_baseline),
            'rms_deviation': float(rms_deviation),
            'guide_y_normalized': float(guide_y),
            'guide_y_px': guide_y_px,
            'search_region': {
                'x_range_px': (x_min_px, x_max_px),
                'y_range_px': (y_min_search, y_max_search)
            },
            'horizontal_bounds': self.horizontal_bounds,
            'vertical_bounds': self.vertical_bounds,
            'num_points': len(xs),
        }

        return xs, ys_smooth, metadata

    def _detect_guide_line(self, img_bgr: np.ndarray, img_path: Path) -> float:
        """
        Detect horizontal line to guide curve search.

        Returns:
            Normalized y-position (0-1) of detected line
        """
        # H = img_bgr.shape[0]

        # Use line detector with vertical bounds
        result = self.line_detector.detect(
            image_path=img_path,
            top_exclusion=self.vertical_bounds[0],
            bottom_exclusion=1.0 - self.vertical_bounds[1]
        )

        # Get horizontal lines
        h_lines = result['horizontal_lines']['lines']

        if h_lines:
            # Use topmost line as guide
            # guide_y = min(line['y_position'] for line in h_lines)

            # Use longest line as guide
            guide_y = max(h_lines, key=lambda line: line['x_length_frac'])['y_position']

            # # Use line closest to center
            # center_y = (self.vertical_bounds[0] + self.vertical_bounds[1]) / 2
            # guide_y = min(h_lines, key=lambda line: abs(line['y_position'] - center_y))['y_position']
        else:
            # Fallback: use middle of vertical bounds
            guide_y = (self.vertical_bounds[0] + self.vertical_bounds[1]) / 2

        return guide_y

    def _trace_columns(self,
                       edges: np.ndarray,
                       x_min: int,
                       x_max: int,
                       y_min: int,
                       y_max: int,
                       guide_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace curve by finding edge points in each column.

        Returns:
            Tuple of (x_array, y_array)
        """
        # H = edges.shape[0]
        xs = np.arange(x_min, x_max + 1, dtype=int)
        ys = []

        search_height = y_max - y_min + 1

        # Create Gaussian weights favoring points near guide
        sigma = search_height / 6
        y_range = np.arange(y_min, y_max + 1)
        weights = np.exp(-0.5 * ((y_range - guide_y) / sigma) ** 2)

        for x in xs:
            col = edges[y_min:y_max + 1, x]

            if not np.any(col):
                # No edges found, use guide
                ys.append(float(guide_y))
                continue

            # Find edge pixels
            edge_indices = np.flatnonzero(col)

            if len(edge_indices) == 0:
                ys.append(float(guide_y))
                continue

            # Get first edge from top (air->liquid transition)
            y_top = y_min + edge_indices[0]

            if len(edge_indices) > 3:
                # Weighted centroid approach
                edge_weights = (col > 0).astype(np.float32) * weights
                weighted_sum = np.sum(edge_weights * y_range)
                total_weight = np.sum(edge_weights) + 1e-6
                y_centroid = weighted_sum / total_weight

                # Blend top edge and centroid
                y_pick = 0.65 * y_top + 0.35 * y_centroid
            else:
                y_pick = float(y_top)

            ys.append(y_pick)

        return xs, np.array(ys, dtype=np.float32)

    def _smooth_curve(self, ys: np.ndarray) -> np.ndarray:
        """
        Apply multi-stage smoothing to curve.

        Args:
            ys: Raw y-coordinates

        Returns:
            Smoothed y-coordinates
        """
        # Rolling median
        ys_smooth = self._rolling_median(ys, self.median_kernel)

        # Hampel filter (remove outliers)
        ys_smooth = self._hampel_filter(ys_smooth, k=7, n_sigma=3.0)

        # Remove short spikes
        ys_smooth = self._remove_short_spikes(ys_smooth, max_run=3, jump_px=8)

        # Enforce continuity
        ys_smooth = self._enforce_continuity(ys_smooth, self.max_step_px)

        return ys_smooth

    def _rolling_median(self, y: np.ndarray, k: int) -> np.ndarray:
        """Apply rolling median filter."""
        k = max(3, k | 1)  # Ensure odd
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        out = np.empty_like(y, dtype=np.float32)
        for i in range(len(y)):
            out[i] = np.median(ypad[i:i + k])
        return out

    def _hampel_filter(self, y: np.ndarray, k: int = 7, n_sigma: float = 3.0) -> np.ndarray:
        """Hampel filter for outlier removal."""
        k = max(3, k | 1)
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode='edge')
        out = y.copy()
        for i in range(len(y)):
            window = ypad[i:i + k]
            median = np.median(window)
            mad = np.median(np.abs(window - median)) + 1e-6
            if abs(y[i] - median) > n_sigma * 1.4826 * mad:
                out[i] = median
        return out

    def _remove_short_spikes(self, y: np.ndarray, max_run: int = 3, jump_px: int = 8) -> np.ndarray:
        """Remove short runs that deviate significantly."""
        med = self._rolling_median(y, k=9)
        out = y.copy()
        i = 0
        N = len(y)
        while i < N:
            if abs(y[i] - med[i]) > jump_px:
                j = i
                while j + 1 < N and abs(y[j + 1] - med[j + 1]) > jump_px:
                    j += 1
                if (j - i + 1) <= max_run:
                    out[i:j + 1] = med[i:j + 1]
                i = j + 1
            else:
                i += 1
        return out

    def _enforce_continuity(self, y: np.ndarray, max_step: int) -> np.ndarray:
        """Enforce maximum step size between adjacent points."""
        out = y.copy()
        for i in range(1, len(y)):
            diff = out[i] - out[i - 1]
            if diff > max_step:
                out[i] = out[i - 1] + max_step
            elif diff < -max_step:
                out[i] = out[i - 1] - max_step
        return out

    def visualize(self,
                  img_bgr: np.ndarray,
                  xs: np.ndarray,
                  ys: np.ndarray,
                  metadata: dict,
                  output_path: Path):
        """
        Create visualization of traced curve.

        Args:
            img_bgr: Original image
            xs: X-coordinates of curve
            ys: Y-coordinates of curve
            metadata: Metadata from tracing
            output_path: Path to save visualization
        """
        overlay = img_bgr.copy()
        H, W = img_bgr.shape[:2]

        # Draw search region
        x_min, x_max = metadata['search_region']['x_range_px']
        y_min, y_max = metadata['search_region']['y_range_px']
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1)

        # Draw guide line
        guide_y = metadata['guide_y_px']
        cv2.line(overlay, (x_min, guide_y), (x_max, guide_y), (255, 0, 255), 1, cv2.LINE_AA)

        # Draw traced curve
        for i in range(1, len(xs)):
            pt1 = (int(xs[i - 1]), int(ys[i - 1]))
            pt2 = (int(xs[i]), int(ys[i]))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw boundary lines
        top_y = int(self.vertical_bounds[0] * H)
        bottom_y = int(self.vertical_bounds[1] * H)
        cv2.line(overlay, (0, top_y), (W, top_y), (255, 0, 0), 1)
        cv2.line(overlay, (0, bottom_y), (W, bottom_y), (255, 0, 0), 1)

        left_x = int(self.horizontal_bounds[0] * W)
        right_x = int(self.horizontal_bounds[1] * W)
        cv2.line(overlay, (left_x, 0), (left_x, H), (255, 0, 0), 1)
        cv2.line(overlay, (right_x, 0), (right_x, H), (255, 0, 0), 1)

        # Add text info
        cv2.putText(overlay, f"Points: {len(xs)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Guide: y={metadata['guide_y_normalized']:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Variance={np.var(ys):.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save
        cv2.imwrite(str(output_path), overlay)


def main():
    parser = argparse.ArgumentParser(description="Guided curve tracing for liquid-air interface")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-o", "--outdir", default="guided_curve_results", help="Output directory")
    parser.add_argument("--guide-y", type=float, default=None,
                        help="Optional guide y-position (0-1). If not provided, will detect automatically")
    parser.add_argument("--top", type=float, default=0.25, help="Top boundary (fraction)")
    parser.add_argument("--bottom", type=float, default=0.90, help="Bottom boundary (fraction)")
    parser.add_argument("--left", type=float, default=0.05, help="Left boundary (fraction)")
    parser.add_argument("--right", type=float, default=0.95, help="Right boundary (fraction)")
    parser.add_argument("--search-offset", type=int, default=30,
                        help="Vertical search offset around guide line (pixels)")
    parser.add_argument("--median-k", type=int, default=9, help="Median filter kernel size")
    parser.add_argument("--max-step", type=int, default=4, help="Max step between points (pixels)")

    args = parser.parse_args()

    # image path
    img_path = Path(args.image)

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")

    # Initialize tracer
    tracer = GuidedCurveTracer(
        vertical_bounds=(args.top, args.bottom),
        horizontal_bounds=(args.left, args.right),
        search_offset_px=args.search_offset,
        median_kernel=args.median_k,
        max_step_px=args.max_step
    )

    # Trace curve
    print("Tracing liquid-air interface curve...")
    xs, ys, metadata = tracer.trace_curve(img, img_path, guide_y=args.guide_y)

    # Print results
    print(f"\nResults:")
    print(f"  Points traced: {len(xs)}")
    print(f"  Guide y-position: {metadata['guide_y_normalized']:.3f}")
    print(f"  Y-range: [{ys.min():.1f}, {ys.max():.1f}] px")
    print(f"  Y-variance: {np.var(ys):.2f}")

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save visualization
    stem = Path(args.image).stem
    viz_path = outdir / f"{stem}_guided_curve.png"
    tracer.visualize(img, xs, ys, metadata, viz_path)
    print(f"\nVisualization saved to: {viz_path}")

    # Save curve data
    data_path = outdir / f"{stem}_curve_data.json"
    # save as json
    with open(data_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Curve data saved to: {data_path}")


if __name__ == "__main__":
    main()