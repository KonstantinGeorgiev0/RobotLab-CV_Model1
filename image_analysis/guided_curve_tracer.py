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
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_analysis.line_hv_detection import LineDetector
from config import CURVE_PARAMS


class GuidedCurveTracer:
    """Traces curved liquid-air interface using horizontal line detection as guide."""

    def __init__(self,
                 vertical_bounds: Tuple[float, float] = (0.20, 0.80),
                 horizontal_bounds: Tuple[float, float] = (0.05, 0.95),
                 search_offset_frac: float = 0.10,
                 median_kernel: int = 9,
                 max_step_px: int = 4,
                 ):
        """
        Initialize guided curve tracer.

        Args:
            vertical_bounds: (top_frac, bottom_frac) - fraction of image height
            horizontal_bounds: (left_frac, right_frac) - fraction of image width
            search_offset_frac: Vertical offset around detected horizontal line to search
            median_kernel: Kernel size for median smoothing
            max_step_px: Maximum allowed step between adjacent points
        """
        self.vertical_bounds = vertical_bounds
        self.horizontal_bounds = horizontal_bounds
        self.search_offset_frac = search_offset_frac
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
            guide_y = self._detect_guide_line(img_bgr, img_path, H, W)

        # Convert guide to pixel coordinates
        guide_y_px = int(guide_y * H)

        # Define search region
        x_min_px = int(self.horizontal_bounds[0] * W)
        x_max_px = int(self.horizontal_bounds[1] * W)

        # Define search region
        search_offset_px = int(self.search_offset_frac * H)
        y_min_search = max(0, guide_y_px - search_offset_px)
        y_max_search = min(H - 1, guide_y_px + search_offset_px)

        # Extract edges
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Define search region
        search_roi = gray[y_min_search:y_max_search + 1, x_min_px:x_max_px + 1]
        # Apply blur for noise reduction
        search_roi = cv2.GaussianBlur(search_roi, (3, 3), 0)
        # Contrast for low-light vials
        search_roi = cv2.equalizeHist(search_roi)
        # Vertical gradient using Sobel operator
        sobel_y = cv2.Sobel(search_roi, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.abs(sobel_y)  # Magnitude
        edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)  # Normalize
        thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # save edges for debugging
        cv2.imwrite(str(img_path.parent.parent.parent / Path(str("image_analysis")) / Path(str("guided_curve_results"))
                        / "edges" / f'edges_{img_path.stem}.png'), edges)
        # save image with detected edges
        cv2.imwrite(str(img_path.parent.parent.parent / Path(str("image_analysis")) / Path(str("guided_curve_results"))
                        / "edges" / f'edges_detected_{img_path.stem}.png'), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

        # Trace curve column by column
        xs, ys = self._trace_columns_topmost_prefer(
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


    def _detect_guide_line(self, img_bgr: np.ndarray, img_path: Path, height: int, width:int) -> float:
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
            # Weighted average of lines to get guide y
            total_weight = sum(l.x_length_frac for l in h_lines)
            guide_y = sum(l.x_length_frac * l.y_position for l in h_lines) / total_weight if total_weight > 0 \
                else (self.vertical_bounds[0] + self.vertical_bounds[1]) / 2

        else:
            # scan central column for max gradient / strongest transition
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            center_x = width // 2
            col_grad = cv2.Sobel(gray[:, center_x], cv2.CV_64F, 0, 1, ksize=5)
            y_min_px = int(self.vertical_bounds[0] * height)
            y_max_px = int(self.vertical_bounds[1] * height)
            guide_y_px = y_min_px + np.argmax(np.abs(col_grad[y_min_px:y_max_px]))
            guide_y = guide_y_px / height

        return guide_y

    def _trace_columns_smoother(self,
                       edges: np.ndarray,
                       x_min: int,
                       x_max: int,
                       y_min: int,
                       y_max: int,
                       guide_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace curve by finding edge points in each column and resulting in smoother curved lines.

        Returns:
            Tuple of (x_array, y_array)
        """
        search_height = y_max - y_min + 1
        if search_height <= 0:
            return np.array([]), np.array([])

        xs = np.arange(x_min, x_max + 1, dtype=int)
        ys = np.full(len(xs), float(guide_y), dtype=np.float32)  # Default to guide

        # Gaussian weights
        y_range = np.arange(y_min, y_max + 1)
        sigma = search_height / 4.0  # Balanced sigma
        weights = np.exp(-0.5 * ((y_range - guide_y) / sigma) ** 2)

        min_strength = 5  # Ignore very weak pixels

        for i, x in enumerate(xs):
            col = edges[:, x - x_min].astype(np.float32)  # Magnitude for scoring

            # Find candidate edges
            edge_indices = np.flatnonzero(col > min_strength)
            if len(edge_indices) == 0:
                continue

            # Scores
            scores = col[edge_indices] * weights[edge_indices]

            if len(edge_indices) <= 3:
                # Pick strongest scored
                best_idx = np.argmax(scores)
                y_pick = y_min + edge_indices[best_idx]
            else:
                # Weighted centroid
                weighted_sum = np.sum(scores * (y_min + edge_indices))
                total_weight = np.sum(scores) + 1e-6
                y_pick = weighted_sum / total_weight

            ys[i] = y_pick

        return xs, ys

    def _trace_columns_topmost_prefer(self,
                       edges: np.ndarray,
                       x_min: int,
                       x_max: int,
                       y_min: int,
                       y_max: int,
                       guide_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace curve by finding edge points in each column, preferring topmost strong edges when multiple distant ones exist.

        Returns:
            Tuple of (x_array, y_array)
        """
        search_height = y_max - y_min + 1
        if search_height <= 0:
            return np.array([]), np.array([])

        xs = np.arange(x_min, x_max + 1, dtype=int)
        ys = np.full(len(xs), float(guide_y), dtype=np.float32)  # Default to guide

        # Gaussian weights
        y_range = np.arange(y_min, y_max + 1)
        sigma = search_height / 4.0  # Balanced sigma
        weights = np.exp(-0.5 * ((y_range - guide_y) / sigma) ** 2)

        min_strength = 50  # Minimum peak height
        min_distance = 3  # Minimum distance between peaks in px
        separation_thresh = 10  # px threshold for 'distant'
        strength_ratio_thresh = 1.5  # If max/min >= this, not similar

        for i, x in enumerate(xs):
            col = edges[:, x - x_min].astype(np.float32)  # Magnitude for peak detection

            # Find distinct peaks
            peaks, properties = find_peaks(col, height=min_strength, distance=min_distance)
            if len(peaks) == 0:
                continue  # Keep guide_y

            peak_heights = properties['peak_heights']

            # Sort by y-position ascending
            sorted_idx = np.argsort(peaks)
            sorted_peaks = peaks[sorted_idx]
            sorted_heights = peak_heights[sorted_idx]

            # Group close peaks
            groups = []
            current_group_peaks = [sorted_peaks[0]]
            current_group_heights = [sorted_heights[0]]
            for k in range(1, len(sorted_peaks)):
                delta_y = sorted_peaks[k] - sorted_peaks[k - 1]
                if delta_y <= separation_thresh:
                    current_group_peaks.append(sorted_peaks[k])
                    current_group_heights.append(sorted_heights[k])
                else:
                    # End current group
                    group_top_y = min(current_group_peaks)
                    group_max_h = max(current_group_heights)
                    groups.append((group_top_y, group_max_h))
                    current_group_peaks = [sorted_peaks[k]]
                    current_group_heights = [sorted_heights[k]]

            # Add last group
            if current_group_peaks:
                group_top_y = min(current_group_peaks)
                group_max_h = max(current_group_heights)
                groups.append((group_top_y, group_max_h))

            # filter distant groups if similar strength
            keep_groups = [groups[0]]  # Start with top group
            j = 0
            for k in range(1, len(groups)):
                delta_y = groups[k][0] - groups[j][0]
                max_h = max(groups[k][1], groups[j][1])
                min_h = min(groups[k][1], groups[j][1])
                ratio = max_h / min_h

                if delta_y > separation_thresh and ratio < strength_ratio_thresh:
                    # Distant and similar
                    pass
                else:
                    # Keep
                    keep_groups.append(groups[k])
                    j = k

            # score kept groups
            if keep_groups:
                group_ys = [g[0] for g in keep_groups]
                group_hs = [g[1] for g in keep_groups]
                # weights indexed by relative y
                scores = [h * weights[y + y_min - y_min] for y, h in zip(group_ys, group_hs)]  # y is relative
                best_group_idx = np.argmax(scores)
                y_pick = y_min + group_ys[best_group_idx]
                ys[i] = y_pick

        return xs, ys


    def _trace_columns_basic(self,
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
        H = edges.shape[0]
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

            # Get first edge from top
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
        ys_smooth = self._remove_short_spikes(ys_smooth, max_run=2, jump_px=2)

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
        cv2.putText(overlay, f"Variance: {metadata['y_variance']:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save
        cv2.imwrite(str(output_path), overlay)


def main():
    parser = argparse.ArgumentParser(description="Guided curve tracing for liquid-air interface")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-o", "--outdir", default="guided_curve_results", help="Output directory")
    parser.add_argument("--guide-y", type=float, default=None,
                        help="Optional guide y-position (0-1). If not provided, will detect automatically")
    parser.add_argument("--top", type=float, default=0.30, help="Top boundary (fraction)")
    parser.add_argument("--bottom", type=float, default=0.80, help="Bottom boundary (fraction)")
    parser.add_argument("--left", type=float, default=0.05, help="Left boundary (fraction)")
    parser.add_argument("--right", type=float, default=0.95, help="Right boundary (fraction)")
    parser.add_argument("--search-offset", type=float, default=0.10,
                        help="Vertical search offset around guide line (fraction of image height)")
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
        search_offset_frac=args.search_offset,
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
    stem = Path(args.image).stem
    outdir = Path(args.outdir) / f"{stem}_result"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save visualization
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