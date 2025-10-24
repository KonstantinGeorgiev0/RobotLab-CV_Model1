"""
Variance analysis for guided curve tracer with comprehensive analytics.
Provides detailed statistical analysis and visualization tools.
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, interpolate
from scipy import stats as scipy_stats
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from image_analysis.guided_curve_tracer import GuidedCurveTracer

@dataclass
class CurveStatistics:
    """Comprehensive statistics for curve analysis."""
    # Basic statistics
    mean_y: float
    median_y: float
    std_dev: float
    variance: float
    range: float

    # Deviation from baseline
    baseline_y: float
    variance_from_baseline: float
    std_dev_from_baseline: float
    rms_deviation: float
    inter_segment_variance: float
    mad: float  # Median Absolute Deviation

    # Distribution statistics
    skewness: float
    kurtosis: float

    # Trend analysis
    linear_trend_slope: float
    linear_trend_r_squared: float
    residual_variance: float
    detrended_std: float

    # Local characteristics
    roughness: float  # Second derivative measure
    local_variance_mean: float
    local_variance_std: float

    # Curvature metrics
    max_curvature: float
    mean_curvature: float
    curvature_changes: int

    # Quality metrics
    num_points: int
    curve_width: float
    point_density: float  # Points per pixel width

    # Frequency analysis
    dominant_frequency: float
    spectral_energy: float


class CurveAnalyzer:
    """Analysis tools for guided curve tracer."""

    def __init__(self):
        pass

    def compute_comprehensive_statistics(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            baseline_y: Optional[float] = None,
            inter_segment_variance: Optional[float] = None,
            window_size: int = 20
    ) -> CurveStatistics:
        """
        Compute comprehensive statistical analysis of curve.

        Args:
            xs: X-coordinates
            ys: Y-coordinates
            baseline_y: Reference line (if None, uses median)
            window_size: Window for local variance analysis

        Returns:
            CurveStatistics object with all metrics
        """
        if len(ys) == 0:
            return self._empty_statistics()

        # Baseline
        if baseline_y is None:
            baseline_y = np.median(ys)

        # Basic statistics
        mean_y = float(np.mean(ys))
        median_y = float(np.median(ys))
        std_dev = float(np.std(ys))
        variance = float(np.var(ys))
        y_range = float(np.ptp(ys))

        # Deviation from baseline
        deviations = ys - baseline_y
        variance_from_baseline = float(np.var(deviations))
        std_dev_from_baseline = float(np.std(deviations))
        rms_deviation = float(np.sqrt(np.mean(deviations ** 2)))
        mad = float(np.median(np.abs(deviations)))

        # Distribution statistics
        skewness = float(scipy_stats.skew(ys))
        kurtosis = float(scipy_stats.kurtosis(ys))

        # Trend analysis
        linear_stats = self._analyze_linear_trend(xs, ys)

        # Local characteristics
        roughness = self._compute_roughness(ys)
        local_var_mean, local_var_std = self._compute_local_variance(ys, window_size)

        # Curvature analysis
        curvature_stats = self._analyze_curvature(xs, ys)

        # Fourier metrics
        curve_width = float(xs[-1] - xs[0]) if len(xs) > 1 else 0.0
        point_density = len(xs) / curve_width if curve_width > 0 else 0.0

        # Frequency analysis
        freq_stats = self._analyze_frequency(ys)

        return CurveStatistics(
            mean_y=mean_y,
            median_y=median_y,
            std_dev=std_dev,
            variance=variance,
            range=y_range,
            baseline_y=float(baseline_y),
            variance_from_baseline=variance_from_baseline,
            std_dev_from_baseline=std_dev_from_baseline,
            rms_deviation=rms_deviation,
            inter_segment_variance=inter_segment_variance if inter_segment_variance is not None else 0.0,
            mad=mad,
            skewness=skewness,
            kurtosis=kurtosis,
            linear_trend_slope=linear_stats['slope'],
            linear_trend_r_squared=linear_stats['r_squared'],
            residual_variance=linear_stats['residual_variance'],
            detrended_std=linear_stats['detrended_std'],
            roughness=roughness,
            local_variance_mean=local_var_mean,
            local_variance_std=local_var_std,
            max_curvature=curvature_stats['max_curvature'],
            mean_curvature=curvature_stats['mean_curvature'],
            curvature_changes=curvature_stats['curvature_changes'],
            num_points=len(xs),
            curve_width=curve_width,
            point_density=point_density,
            dominant_frequency=freq_stats['dominant_frequency'],
            spectral_energy=freq_stats['spectral_energy'],
        )

    def _analyze_linear_trend(self, xs: np.ndarray, ys: np.ndarray) -> Dict:
        """Analyze linear trend in curve."""
        if len(xs) < 3:
            return {
                'slope': 0.0,
                'r_squared': 0.0,
                'residual_variance': 0.0,
                'detrended_std': 0.0
            }

        # Fit linear trend
        coeffs = np.polyfit(xs, ys, deg=1)
        y_fit = np.polyval(coeffs, xs)

        # Residuals
        residuals = ys - y_fit
        residual_variance = float(np.var(residuals))
        detrended_std = float(np.std(residuals))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            'slope': float(coeffs[0]),
            'r_squared': r_squared,
            'residual_variance': residual_variance,
            'detrended_std': detrended_std
        }

    def _compute_roughness(self, ys: np.ndarray) -> float:
        """Compute surface roughness (second derivative measure)."""
        if len(ys) < 3:
            return 0.0

        # Approximate second derivative
        d2y = np.diff(ys, n=2)
        return float(np.std(d2y))

    def _compute_local_variance(self, ys: np.ndarray, window_size: int) -> Tuple[float, float]:
        """Compute local variance statistics."""
        if len(ys) < window_size:
            return 0.0, 0.0

        local_variances = []
        for i in range(len(ys) - window_size + 1):
            window = ys[i:i + window_size]
            local_variances.append(np.var(window))

        return float(np.mean(local_variances)), float(np.std(local_variances))


    def _analyze_curvature(self, xs: np.ndarray, ys: np.ndarray) -> Dict:
        """Analyze curvature of the curve."""
        if len(xs) < 5:
            return {
                'max_curvature': 0.0,
                'mean_curvature': 0.0,
                'curvature_changes': 0
            }

        # Compute first and second derivatives
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2) + 1e-8
        curvature = numerator / denominator

        # Detect sign changes in curvature (inflection points)
        d_curvature = np.diff(curvature)
        curvature_changes = np.sum(np.abs(np.diff(np.sign(d_curvature))) > 0)

        return {
            'max_curvature': float(np.max(curvature)),
            'mean_curvature': float(np.mean(curvature)),
            'curvature_changes': int(curvature_changes)
        }


    def _analyze_frequency(self, ys: np.ndarray) -> Dict:
        """Perform frequency domain analysis."""
        if len(ys) < 10:
            return {
                'dominant_frequency': 0.0,
                'spectral_energy': 0.0,
                'fourier_classification': 'insufficient_data'
            }

        # Remove mean (DC component)
        ys_centered = ys - np.mean(ys)

        # Compute FFT
        fft = np.fft.rfft(ys_centered)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(ys_centered))

        # Compute frequency bands
        total_power = np.sum(power)

        # Find dominant frequency (excluding DC)
        if len(power) > 1:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_freq = float(freqs[dominant_idx])
        else:
            dominant_freq = 0.0

        # Total spectral energy
        spectral_energy = float(total_power)

        return {
            'dominant_frequency': dominant_freq,
            'spectral_energy': spectral_energy,
        }


    def _empty_statistics(self) -> CurveStatistics:
        """Return empty statistics object."""
        return CurveStatistics(
            mean_y=0.0, median_y=0.0, std_dev=0.0, variance=0.0, range=0.0,
            baseline_y=0.0, variance_from_baseline=0.0, std_dev_from_baseline=0.0,
            rms_deviation=0.0, inter_segment_variance=0.0, mad=0.0, skewness=0.0, kurtosis=0.0,
            linear_trend_slope=0.0, linear_trend_r_squared=0.0,
            residual_variance=0.0, detrended_std=0.0, roughness=0.0,
            local_variance_mean=0.0, local_variance_std=0.0,
            max_curvature=0.0, mean_curvature=0.0, curvature_changes=0,
            num_points=0, curve_width=0.0, point_density=0.0,
            dominant_frequency=0.0, spectral_energy=0.0
        )

    def detect_anomalies(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            threshold_sigma: float = 3.0
    ) -> Dict:
        """
        Detect anomalous regions in the curve.

        Args:
            xs: X-coordinates
            ys: Y-coordinates
            threshold_sigma: Number of standard deviations for anomaly detection

        Returns:
            Dictionary with anomaly information
        """
        if len(ys) < 10:
            return {'anomaly_indices': [], 'num_anomalies': 0}

        # Fit polynomial trend
        coeffs = np.polyfit(xs, ys, deg=3)
        y_trend = np.polyval(coeffs, xs)

        # Compute residuals
        residuals = ys - y_trend

        # Detect outliers
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))

        # Modified Z-score
        modified_z_scores = 0.6745 * (residuals - median_residual) / (mad + 1e-8)

        anomaly_mask = np.abs(modified_z_scores) > threshold_sigma
        anomaly_indices = np.where(anomaly_mask)[0].tolist()

        return {
            'anomaly_indices': anomaly_indices,
            'num_anomalies': len(anomaly_indices),
            'anomaly_positions': [(int(xs[i]), float(ys[i])) for i in anomaly_indices],
            'max_deviation': float(np.max(np.abs(residuals))) if len(residuals) > 0 else 0.0
        }

    def compute_smoothness_score(self, ys: np.ndarray) -> float:
        """
        Compute smoothness score (0-100, higher is smoother).

        Based on second derivative magnitude.
        """
        if len(ys) < 3:
            return 100.0

        # Compute second derivative
        d2y = np.diff(ys, n=2)
        roughness = np.std(d2y)

        # Convert to score (inverse relationship)
        score = 100.0 * np.exp(-roughness / 2.0)

        return float(np.clip(score, 0.0, 100.0))

    def segment_curve(
            self,
            xs: np.ndarray,
            ys: np.ndarray,
            num_segments: int = 5
    ) -> Dict:
        """
        Segment curve into regions and analyze each.

        Args:
            xs: X-coordinates
            ys: Y-coordinates
            num_segments: Number of segments to divide curve into

        Returns:
            Dictionary with segment statistics
        """
        if len(xs) < num_segments * 2:
            num_segments = max(1, len(xs) // 2)

        segment_size = len(xs) // num_segments
        segments = []

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(xs)

            xs_seg = xs[start_idx:end_idx]
            ys_seg = ys[start_idx:end_idx]

            if len(ys_seg) > 0:
                segment_stats = {
                    'segment_id': i,
                    'x_range': (float(xs_seg[0]), float(xs_seg[-1])),
                    'mean_y': float(np.mean(ys_seg)),
                    'std_dev': float(np.std(ys_seg)),
                    'variance': float(np.var(ys_seg)),
                    'range': float(np.ptp(ys_seg)),
                    'num_points': len(ys_seg)
                }
                segments.append(segment_stats)

        # Compute segment-to-segment variance
        segment_means = [s['mean_y'] for s in segments]
        inter_segment_variance = float(np.var(segment_means))

        return {
            'segments': segments,
            'num_segments': len(segments),
            'inter_segment_variance': inter_segment_variance,
            'max_segment_std': max([s['std_dev'] for s in segments]) if segments else 0.0
        }


class CurveVisualizer:
    """Visualization for curve analysis."""

    def __init__(self):
        self.analyzer = CurveAnalyzer()

    def create_image_overlay(
            self,
            img: np.ndarray,
            xs: np.ndarray,
            ys: np.ndarray,
            stats: CurveStatistics,
            curve_metadata: Dict,
            anomalies: Dict,
            output_path: Path
    ):
        """Save only the original image with curve & overlays."""
        H, W = img.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Plot anomalies first
        if anomalies['num_anomalies'] > 0:
            # Extract x,y coordinates from anomaly indices
            anomaly_indices = anomalies['anomaly_indices']
            anomaly_x = xs[anomaly_indices]
            anomaly_y = ys[anomaly_indices]
            ax.plot(anomaly_x, anomaly_y, 'x', color='red',
                    markersize=10, markeredgewidth=2, label='Anomalies', zorder=3)

        # Plot detected curve
        ax.plot(xs, ys, color='green', linewidth=1.5, label='Detected Curve', zorder=5)

        # Baseline
        if stats.baseline_y is not None:
            ax.axhline(y=stats.baseline_y, color='magenta', linestyle='--', linewidth=1, alpha=0.8, label='Baseline')

        # Search region
        if 'search_region' in curve_metadata:
            sr = curve_metadata['search_region']
            y_min, y_max = sr['y_range_px']
            x_min, x_max = sr['x_range_px']
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 fill=False, edgecolor='yellow', linewidth=1.2, alpha=0.7)
            ax.add_patch(rect)

        ax.set_title(f'Curve Overlay (n={len(xs)})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Image overlay saved to: {output_path}")


    def create_analytics_plot(
            self,
            img: np.ndarray,
            xs: np.ndarray,
            ys: np.ndarray,
            stats: CurveStatistics,
            curve_metadata: Dict,
            anomalies: Dict,
            segments: Dict,
            output_path: Path
    ):
        """Save only the analytics panels."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # === PANEL A: Text stats
        ax2 = fig.add_subplot(gs[0, 0])
        ax2.axis('off')
        smoothness = self.analyzer.compute_smoothness_score(ys)
        summary_text = f"""
                CURVE STATISTICS
                {'=' * 35}
                
                POSITION:
                  Mean Y: {stats.mean_y:.2f} px
                  Median Y: {stats.median_y:.2f} px
                  Baseline Y: {stats.baseline_y:.2f} px
                  Range: {stats.range:.2f} px
                
                VARIANCE:
                  Variance: {stats.variance_from_baseline:.4f}
                  Std Dev: {stats.std_dev_from_baseline:.2f} px
                  RMS Dev: {stats.rms_deviation:.2f} px
                  MAD: {stats.mad:.2f} px
                
                QUALITY:
                  Smoothness: {smoothness:.1f}/100
                  R² (linear): {stats.linear_trend_r_squared:.4f}
                  Roughness: {stats.roughness:.2f}
                
                SHAPE:
                  Trend slope: {stats.linear_trend_slope:.6f}
                  Max curvature: {stats.max_curvature:.4f}
                  Inflections: {stats.curvature_changes}
                
                ANOMALIES:
                  Detected: {anomalies.get('num_anomalies', 0)}
                  Max deviation: {anomalies.get('max_deviation', 0.0):.2f} px
                """
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                 fontfamily='monospace', fontsize=9, verticalalignment='top')

        # === PANEL B: Deviation from baseline
        ax3 = fig.add_subplot(gs[0, 1])
        deviations = ys - stats.baseline_y
        ax3.plot(xs, deviations, linewidth=1.5, alpha=0.8)
        ax3.fill_between(xs, 0, deviations, alpha=0.25)
        ax3.axhline(0, color='black', linewidth=1)
        ax3.axhline(stats.std_dev_from_baseline, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax3.axhline(-stats.std_dev_from_baseline, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax3.set_xlabel('X (px)')
        ax3.set_ylabel('Deviation (px)')
        ax3.set_title('Deviation from Baseline')
        ax3.grid(True, alpha=0.3)

        # === PANEL C: Histogram
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.hist(deviations, bins=30, density=True, alpha=0.7, edgecolor='black')
        mu, sigma = 0, stats.std_dev_from_baseline
        x_norm = np.linspace(deviations.min(), deviations.max(), 100)
        ax4.plot(x_norm, scipy_stats.norm.pdf(x_norm, mu, sigma), linewidth=2)
        ax4.set_xlabel('Deviation (px)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Distribution (skew={stats.skewness:.2f}, kurt={stats.kurtosis:.2f})')
        ax4.grid(True, alpha=0.3)

        # === PANEL D: Curvature
        ax5 = fig.add_subplot(gs[1, 0])
        if len(xs) >= 5:
            dx = np.gradient(xs)
            dy = np.gradient(ys)
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            curvature = np.abs(dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** (3 / 2) + 1e-8)
            ax5.plot(xs, curvature, linewidth=1.5)
            ax5.fill_between(xs, 0, curvature, alpha=0.3)
            ax5.set_xlabel('X (px)')
            ax5.set_ylabel('Curvature')
            ax5.set_title(f'Curvature (max={stats.max_curvature:.4f})')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Insufficient points', ha='center', va='center', transform=ax5.transAxes)

        # === PANEL E: Local variance
        ax6 = fig.add_subplot(gs[1, 1])
        window_size = 20
        if len(ys) >= window_size:
            local_vars = [np.var(ys[i:i + window_size]) for i in range(len(ys) - window_size + 1)]
            # match local_x to the length of local_vars
            local_x = xs[window_size // 2: len(local_vars) + window_size // 2]
            ax6.plot(local_x, local_vars, linewidth=1.5)
            ax6.axhline(np.mean(local_vars), linestyle='--', linewidth=1, label=f"Mean={np.mean(local_vars):.2f}")
            ax6.set_xlabel('X (px)')
            ax6.set_ylabel('Local variance')
            ax6.set_title(f'Local Variance (window={window_size})')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Insufficient points', ha='center', va='center', transform=ax6.transAxes)

        # === PANEL F: Frequency spectrum
        ax7 = fig.add_subplot(gs[1, 2])
        if len(ys) >= 10:
            ys_dt = signal.detrend(ys)
            fft = np.fft.rfft(ys_dt);
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(ys_dt))
            ax7.semilogy(freqs[1:], power[1:], linewidth=1.5)
            ax7.axvline(x=stats.dominant_frequency, linestyle='--', linewidth=1,
                        label=f'Dominant {stats.dominant_frequency:.4f}')
            ax7.set_xlabel('Frequency')
            ax7.set_ylabel('Power')
            ax7.set_title('Frequency Spectrum')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Insufficient points', ha='center', va='center', transform=ax7.transAxes)

        # === PANEL G: Segment analysis
        ax8 = fig.add_subplot(gs[2, :])
        if segments['num_segments'] > 0:
            seg_ids = [s['segment_id'] for s in segments['segments']]
            seg_means = [s['mean_y'] for s in segments['segments']]
            seg_stds = [s['std_dev'] for s in segments['segments']]
            ax8.errorbar(seg_ids, seg_means, yerr=seg_stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=7)
            ax8.axhline(y=stats.baseline_y, linestyle='--', linewidth=1, label='Baseline')
            ax8.set_xlabel('Segment ID')
            ax8.set_ylabel('Mean Y (px)')
            ax8.set_title(f'Segment Analysis (n={segments["num_segments"]})')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No segment data', ha='center', va='center', transform=ax8.transAxes)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Analytics plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="  variance analysis for guided curve tracer"
    )
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-o", "--outdir", default="guided_curve_analysis",
                        help="Output directory")

    # Curve tracer parameters
    parser.add_argument("--guide-y", type=float, default=None,
                        help="Optional guide y-position (0-1)")
    parser.add_argument("--top", type=float, default=0.25)
    parser.add_argument("--bottom", type=float, default=0.80)
    parser.add_argument("--left", type=float, default=0.05)
    parser.add_argument("--right", type=float, default=0.95)
    parser.add_argument("--search-offset", type=float, default=0.1)
    parser.add_argument("--median-k", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=4)

    # Analysis parameters
    parser.add_argument("--window-size", type=int, default=20,
                        help="Window size for local variance analysis")
    parser.add_argument("--num-segments", type=int, default=5,
                        help="Number of segments for regional analysis")
    parser.add_argument("--anomaly-threshold", type=float, default=3.0,
                        help="Sigma threshold for anomaly detection")

    args = parser.parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    # Create subfolder based on image name
    result_dir = outdir / f"{img_path.stem}_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    H, W = img.shape[:2]

    print(f"\n{'=' * 70}")
    print(f"  GUIDED CURVE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Image: {img_path.name}")
    print(f"Size: {W}x{H}")

    # Initializations
    tracer = GuidedCurveTracer(
        vertical_bounds=(args.top, args.bottom),
        horizontal_bounds=(args.left, args.right),
        search_offset_frac=args.search_offset,
        median_kernel=args.median_k,
        max_step_px=args.max_step
    )
    xs, ys, curve_metadata = tracer.trace_curve(img, img_path, guide_y=args.guide_y)

    analyzer = CurveAnalyzer()
    segments = analyzer.segment_curve(
        xs=np.asarray(xs, dtype=np.float32),
        ys=np.asarray(ys, dtype=np.float32),
        num_segments=args.num_segments,
    )

    curve_stats = analyzer.compute_comprehensive_statistics(
        xs=np.asarray(xs, dtype=np.float32),
        ys=np.asarray(ys, dtype=np.float32),
        baseline_y=None,
        inter_segment_variance=segments['inter_segment_variance'],
        window_size=args.window_size,
    )

    anomalies = analyzer.detect_anomalies(
        xs=np.asarray(xs, dtype=np.float32),
        ys=np.asarray(ys, dtype=np.float32),
        threshold_sigma=args.anomaly_threshold,
    )

    # Trace curve
    print(f"\n{'=' * 70}")
    print("STEP 1: CURVE DETECTION")
    print(f"{'=' * 70}")
    print(f"Detected {len(xs)} points")
    print(f"Guide Y: {curve_metadata.get('guide_y_normalized', float('nan')):.3f} "
          f"({curve_metadata.get('guide_y_px', 'N/A')} px)")
    if 'search_region' in curve_metadata:
        sr = curve_metadata['search_region']
        print(f"Search region: x[{sr['x_range_px'][0]}, {sr['x_range_px'][1]}], "
              f"y[{sr['y_range_px'][0]}, {sr['y_range_px'][1]}] (px)")
    if curve_metadata.get('trace_status'):
        print(f"Trace status: {curve_metadata['trace_status']}")

    if len(xs) == 0:
        # Nothing to analyze
        minimal_report = {
            "image": img_path.name,
            "size": {"width": int(W), "height": int(H)},
            "points_detected": 0,
            "message": "No curve points detected. Check tracer parameters or input image.",
            "tracer_params": {
                "vertical_bounds": [args.top, args.bottom],
                "horizontal_bounds": [args.left, args.right],
                "search_offset_frac": args.search_offset,
                "median_kernel": args.median_k,
                "max_step_px": args.max_step,
                "guide_y": args.guide_y,
            },
            "curve_metadata": curve_metadata,
        }
        report_path = result_dir / f"{img_path.stem}_report.json"
        with open(report_path, "w") as f:
            json.dump(minimal_report, f, indent=2)
        print(f"\nNo points detected. Minimal report saved to: {report_path}")
        return

    # COMPREHENSIVE STATS
    # =========================
    print(f"\n{'=' * 70}")
    print("STEP 2: COMPREHENSIVE STATISTICS")
    print(f"{'=' * 70}")
    print(f"Variance (from baseline): {curve_stats.variance_from_baseline:.4f}")
    print(f"Std Dev (from baseline): {curve_stats.std_dev_from_baseline:.3f} px")
    print(f"Smoothness score: {analyzer.compute_smoothness_score(np.asarray(ys, dtype=np.float32)):.1f}/100")

    # ANOMALY DETECTION
    # =========================
    print(f"\n{'=' * 70}")
    print("STEP 3: ANOMALY DETECTION")
    print(f"{'=' * 70}")

    print(f"Anomalies detected: {anomalies['num_anomalies']}")
    if anomalies['num_anomalies'] > 0:
        print(f"Max deviation: {anomalies['max_deviation']:.2f} px")

    # SEGMENT ANALYSIS
    # =========================
    print(f"\n{'=' * 70}")
    print("STEP 4: SEGMENT ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Segments: {segments['num_segments']}")
    print(f"Inter-segment variance: {segments['inter_segment_variance']:.4f}")
    print(f"Max segment std: {segments['max_segment_std']:.3f}")

    # SAVE ARTIFACTS
    # =========================
    print(f"\n{'=' * 70}")
    print("STEP 5: SAVING ARTIFACTS")
    print(f"{'=' * 70}")

    # Save curve points as CSV
    curve_csv = result_dir / f"{img_path.stem}_curve.csv"
    np.savetxt(curve_csv, np.column_stack([xs, ys]), delimiter=",", header="x,y", comments="")
    print(f"Curve points saved: {curve_csv}")

    # Save JSON report
    report = {
        "image": img_path.name,
        "size": {"width": int(W), "height": int(H)},
        "points_detected": int(len(xs)),
        "curve_metadata": curve_metadata,
        "statistics": asdict(curve_stats),
        "anomalies": anomalies,
        "segments": segments,
        "tracer_params": {
            "vertical_bounds": [args.top, args.bottom],
            "horizontal_bounds": [args.left, args.right],
            "search_offset_frac": args.search_offset,
            "median_kernel": args.median_k,
            "max_step_px": args.max_step,
            "guide_y": args.guide_y,
        },
    }
    report_path = result_dir / f"{img_path.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {report_path}")

    # VISUALIZATION
    # =========================
    visualizer = CurveVisualizer()

    # guided curve visualization
    guided_curve_path = result_dir / f"{img_path.stem}_guided_curve.png"
    tracer.visualize(img, xs, ys, curve_metadata, guided_curve_path)
    print(f"Guided curve visualization saved: {guided_curve_path}")

    overlay_path = result_dir / f"{img_path.stem}_overlay.png"
    visualizer.create_image_overlay(
        img=img,
        xs=np.asarray(xs, dtype=np.float32),
        ys=np.asarray(ys, dtype=np.float32),
        stats=curve_stats,
        curve_metadata=curve_metadata,
        anomalies=anomalies,
        output_path=overlay_path,
    )

    analytics_path = result_dir / f"{img_path.stem}_analytics.png"
    visualizer.create_analytics_plot(
        img=img,
        xs=np.asarray(xs, dtype=np.float32),
        ys=np.asarray(ys, dtype=np.float32),
        stats=curve_stats,
        curve_metadata=curve_metadata,
        anomalies=anomalies,
        segments=segments,
        output_path=analytics_path,
    )

    print(f"\nDone.\nArtifacts:\n  - {curve_csv}\n  - {report_path}\n  - {overlay_path}\n  - {analytics_path}")


if __name__ == "__main__":
    main()