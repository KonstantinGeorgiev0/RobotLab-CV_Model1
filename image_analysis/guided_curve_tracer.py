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
from scipy.signal import savgol_filter

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
                 search_offset_frac: float = 0.05,
                 median_kernel: int = 9,
                 max_step_px: int = 4):
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
            img_bgr: input BGR image
            img_path: kept for compatibility (used by _detect_guide_line if guide_y is None)
            guide_y: optional normalized guide in [0,1]. If None, detected from image.

        Returns:
            (xs, ys_smooth, metadata)
        """
        assert img_bgr is not None and img_bgr.size > 0, "empty image"
        H, W = img_bgr.shape[:2]

        # 1) guide line (normalized -> pixels)
        if guide_y is None:
            guide_y = self._detect_guide_line(img_bgr, img_path)  # expected to return normalized [0,1]
        guide_y = float(np.clip(guide_y, 0.0, 1.0))
        guide_y_px = int(round(guide_y * H))

        # 2) ROI + search region (intersect vertical ROI with ±offset around guide)
        top_frac, bot_frac = self.vertical_bounds
        left_frac, right_frac = self.horizontal_bounds
        y_roi_top = max(0, int(H * top_frac))
        y_roi_bot = min(H, int(H * bot_frac))
        x_min_px = max(0, int(W * left_frac))
        x_max_px = min(W, int(W * right_frac))
        x_min_px = min(x_min_px, W - 1)
        x_max_px = max(x_min_px + 1, x_max_px)

        search_offset_px = max(2, int(self.search_offset_frac * H))
        y_min_search = max(y_roi_top, guide_y_px - search_offset_px)
        y_max_search = min(y_roi_bot - 1, guide_y_px + search_offset_px)

        # 3) luminance-like channel + mild denoise
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=5)

        # 4) column-wise tracing with polarity + continuity
        xs = np.arange(x_min_px, x_max_px, dtype=np.int32)
        ys = np.zeros_like(xs, dtype=np.float32)

        # starting point clamped to search band
        prev_y = float(np.clip(guide_y_px, y_min_search, y_max_search))
        band = max(2, int(self.search_offset_frac * H))  # local band around prev_y each column
        max_step = max(1, int(self.max_step_px))  # continuity limiter (e.g., 4 px)

        # choose expected edge polarity (bright→dark typical for upper edge; flip if your setup differs)
        want_neg = True

        for i, x in enumerate(xs):
            y0 = int(round(prev_y))
            ymin = max(y_min_search, y0 - band)
            ymax = min(y_max_search, y0 + band)
            if ymax <= ymin:
                ys[i] = prev_y
                continue

            col = gray[ymin:ymax + 1, x].astype(np.float32)

            # vertical gradient: [-1, 0, 1]/2
            dy = np.convolve(col, np.array([-1., 0., 1.], np.float32) * 0.5, mode="same")

            # polarity mask + adaptive magnitude threshold (top 15% strongest in band)
            mask = (dy < 0) if want_neg else (dy > 0)
            thr = np.percentile(np.abs(dy), 85) if dy.size > 0 else 0.0
            cand_idx = np.where(mask & (np.abs(dy) >= thr))[0]

            if cand_idx.size:
                k = int(cand_idx[np.argmax(np.abs(dy[cand_idx]))])
                y_pick = float(ymin + k)
            else:
                # fallback to nearest-to-guide within the band
                y_pick = float(np.clip(y0, ymin, ymax))

            # continuity: limit per-column step
            if i > 0 and abs(y_pick - prev_y) > max_step:
                y_pick = prev_y + np.sign(y_pick - prev_y) * max_step

            ys[i] = y_pick
            prev_y = y_pick

        # 5) de-spike + smooth
        ys_smooth = self._smooth_trace_1d(ys)

        # 6) scoring: robust waviness + also keep legacy stats relative to guide for compatibility
        scores = self._score_waviness(ys_smooth, H)

        # legacy (baseline = guide_y_px), kept for compatibility with existing consumers
        deviations_from_guide = ys_smooth - float(guide_y_px)
        y_variance = float(np.var(ys_smooth))
        variance_from_baseline = float(np.var(deviations_from_guide))
        std_dev_from_baseline = float(np.std(deviations_from_guide))
        rms_deviation = float(np.sqrt(np.mean(deviations_from_guide ** 2)))

        # metadata (preserve old keys + add new robust ones)
        metadata = {
            'y_variance': y_variance,
            'std_dev_from_baseline': std_dev_from_baseline,
            'variance_from_baseline': variance_from_baseline,
            'rms_deviation': rms_deviation,
            'guide_y_normalized': float(guide_y),
            'guide_y_px': int(guide_y_px),
            'search_region': {
                'x_range_px': (int(x_min_px), int(x_max_px)),
                'y_range_px': (int(y_min_search), int(y_max_search)),
            },
            'horizontal_bounds': self.horizontal_bounds,
            'vertical_bounds': self.vertical_bounds,
            'num_points': int(len(xs)),
            'abs_variance': float(np.var(ys_smooth)),
            'resid_rms_px': float(scores['resid_rms_px']),
            'resid_rms_norm': float(scores['resid_rms_norm']),
            'resid_var_px2': float(scores['resid_var_px2']),
            'poly_slope_px_per_col': float(scores['slope_px_per_col']),
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

    def _smooth_trace_1d(self, y: np.ndarray) -> np.ndarray:
        """
        Remove solitary spikes and gently smooth the 1D trace.
        Returns float array, same length as input.
        """
        y = y.astype(np.float32, copy=True)
        n = len(y)
        if n < 7:
            return y

        # median filter to kill 1–2 px outliers
        k_med = min(9, (n // 40) | 1)
        if k_med >= 3:
            pad = k_med // 2
            yp = np.pad(y, (pad,), mode="edge")
            y = np.array([np.median(yp[i - pad:i + pad + 1]) for i in range(pad, len(yp) - pad)], np.float32)

        # Savitzky–Golay smoothing to keep gentle curvature
        try:
            win = max(9, (n // 50) | 1)
            win = min(win, n - (n + 1) % 2 - 1) if n % 2 == 0 else min(win, n)
            win = max(5, win | 1)
            y = savgol_filter(y, window_length=win, polyorder=2, mode="interp").astype(np.float32)
        except Exception:
            # fallback: simple moving average
            win = max(5, (n // 60) | 1)
            pad = win // 2
            yp = np.pad(y, (pad,), mode="edge")
            y = np.convolve(yp, np.ones(win, np.float32) / win, mode="valid").astype(np.float32)

        return y

    def _score_waviness(self, y: np.ndarray, H: int) -> dict:
        """
        Compute robust roughness metrics:
        - detrended residual RMS/VAR on middle of the trace
        - also return the fitted slope for debugging
        """
        y = y.astype(np.float64, copy=False)
        n = len(y)
        if n < 10:
            return dict(resid_rms_px=0.0, resid_rms_norm=0.0, resid_var_px2=0.0, slope_px_per_col=0.0)

        # crop away side walls
        lo = int(0.10 * n)
        hi = int(0.90 * n)
        yc = y[lo:hi]
        xc = np.arange(lo, hi, dtype=np.float64)

        # linear detrend
        a, b = np.polyfit(xc, yc, 1)
        trend = a * xc + b
        resid = yc - trend

        # robust outlier clipping
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-6
        resid = np.clip(resid, med - 3 * mad, med + 3 * mad)

        rms_px = float(np.sqrt(np.mean(resid ** 2)))
        var_px2 = float(np.var(resid))
        rms_n = float(rms_px / max(1.0, H))

        return dict(
            resid_rms_px=rms_px,
            resid_rms_norm=rms_n,
            resid_var_px2=var_px2,
            slope_px_per_col=float(a),
        )

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
    parser.add_argument("--search-offset", type=float, default=0.05,
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
    print(f"  residual_variance_px2: {metadata['resid_var_px2']:.3f}")
    print(f"  resid_rms_px: {metadata['resid_rms_px']:.3f}")
    print(f"  poly_slope_px_per_col: {metadata['poly_slope_px_per_col']:.3f}")

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