#!/usr/bin/env python3
"""
Gel line detection module.
Detects sinuous, discontinuous lines characteristic of gelled liquid states.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import label, binary_dilation


@dataclass
class GelLine:
    """Represents a detected gel-like line."""
    y_position: float  # Normalized y position (0=top, 1=bottom)
    x_start: float  # Normalized x start
    x_end: float  # Normalized x end
    sinuous: float  # Measure of line curvature/irregularity (0-1)
    discontinuity: float  # Measure of gaps in line (0-1, higher = more gaps)
    strength: float  # Line strength/confidence (0-1)
    length_pixels: int  # Length in pixels

    @property
    def x_span(self) -> float:
        """Horizontal span of line."""
        return self.x_end - self.x_start

    @property
    def is_gel_like(self) -> bool:
        """Check if line has gel-like characteristics."""
        return (self.sinuous > 0.15 and
                self.discontinuity > 0.1 and
                self.x_span > 0.2)

    def __repr__(self) -> str:
        return (f"GelLine(y={self.y_position:.3f}, "
                f"span={self.x_span:.3f}, "
                f"sinuous={self.sinuous:.3f}, "
                f"discontinuous={self.discontinuity:.3f})")


class GelLineDetector:
    """Detector for gel-like sinuous and discontinuous lines."""

    def __init__(self,
                 # Morphology parameters
                 adaptive_block: int = 15,
                 adaptive_c: int = -2,
                 horiz_kernel_div: int = 20,

                 # Gel detection thresholds
                 min_line_length: float = 0.25,
                 min_x_span: float = 0.2,
                 min_sinuous: float = 0.15,
                 min_discontinuity: float = 0.1,
                 min_gel_lines: int = 3,

                 # Analysis parameters
                 sinuous_window: int = 15,
                 gap_threshold: int = 5,

                 # Exclusion zones (normalized 0-1, from top/bottom)
                 top_exclusion: float = 0.15,
                 bottom_exclusion: float = 0.15):
        """
        Initialize gel line detector.

        Args:
            adaptive_block: Block size for adaptive threshold
            adaptive_c: Constant for adaptive threshold
            horiz_kernel_div: Divisor for horizontal kernel size
            min_line_length: Minimum line length (fraction of width)
            min_x_span: Minimum horizontal span
            min_sinuous: Minimum sinuous score
            min_discontinuity: Minimum discontinuity score
            min_gel_lines: Minimum gel-like lines for gel state
            sinuous_window: Window size for sinuous calculation
            gap_threshold: Minimum gap size to count as discontinuity
            top_exclusion: Exclude top N% of image (0-1)
            bottom_exclusion: Exclude bottom N% of image (0-1)
        """
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.horiz_kernel_div = horiz_kernel_div

        self.min_line_length = min_line_length
        self.min_x_span = min_x_span
        self.min_sinuous = min_sinuous
        self.min_discontinuity = min_discontinuity
        self.min_gel_lines = min_gel_lines

        self.sinuous_window = sinuous_window
        self.gap_threshold = gap_threshold

        self.top_exclusion = top_exclusion
        self.bottom_exclusion = bottom_exclusion

    def detect(self, image_path: Path, save_intermediates: bool = False) -> Dict[str, Any]:
        """
        Detect gel-like lines in image.

        Args:
            image_path: Path to input image
            save_intermediates: Save intermediate processing steps

        Returns:
            Dictionary with detection results
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': f'Failed to load image: {image_path}'}

        H, W = img.shape[:2]

        # Extract edge map optimized for gel detection
        edge_map = self._extract_gel_edges(img)

        # Save intermediate if requested
        if save_intermediates:
            out_dir = image_path.parent / "gel_detection_intermediates"
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / f"{image_path.stem}_edges.png"), edge_map)

        # Detect lines with gel characteristics
        gel_lines = self._detect_gel_lines(edge_map, H, W)

        # Filter by gel-like properties
        gel_like_lines = [l for l in gel_lines if l.is_gel_like]

        # Determine if image shows gelled state
        is_gelled = len(gel_like_lines) >= self.min_gel_lines

        # Create summary
        summary = self._create_summary(gel_lines, gel_like_lines, H, W, is_gelled)

        return {
            'is_gelled': is_gelled,
            'num_gel_lines': len(gel_like_lines),
            'total_lines': len(gel_lines),
            'lines': [self._line_to_dict(l) for l in gel_lines],
            'gel_like_lines': [self._line_to_dict(l) for l in gel_like_lines],
            'summary': summary,
            'image_size': {'height': H, 'width': W}
        }

    def _extract_gel_edges(self, img: np.ndarray) -> np.ndarray:
        """Extract edge map optimized for detecting gel wave patterns."""
        # Convert to grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply stronger Gaussian blur to merge nearby wave components
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Detect edges using Canny with lower thresholds to catch faint gel waves
        edges = cv2.Canny(enhanced, 15, 70)

        # Apply stronger morphological closing to connect wave segments better
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

        # Dilate more to make waves thicker and more connected
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.dilate(edges, kernel_dilate, iterations=2)

        return edges

    def _detect_gel_lines(self,
                          edge_map: np.ndarray,
                          height: int,
                          width: int) -> List[GelLine]:
        """Detect and analyze wave-like contours for gel characteristics."""
        lines = []

        # Apply exclusion zones (mask out top and bottom regions)
        masked_edges = self._apply_exclusion_zones(edge_map, height)

        # Find connected components (contours)
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Skip very small contours
            if len(contour) < 20:
                continue

            # Analyze this contour for gel characteristics
            gel_line = self._analyze_contour_for_gel(contour, height, width)

            if gel_line is not None:
                lines.append(gel_line)

        return lines

    def _apply_exclusion_zones(self, edge_map: np.ndarray, height: int) -> np.ndarray:
        """Apply exclusion zones to mask out top and bottom regions."""
        masked = edge_map.copy()

        # Calculate exclusion boundaries
        top_boundary = int(height * self.top_exclusion)
        bottom_boundary = int(height * (1 - self.bottom_exclusion))

        # Mask out top region
        if top_boundary > 0:
            masked[:top_boundary, :] = 0

        # Mask out bottom region
        if bottom_boundary < height:
            masked[bottom_boundary:, :] = 0

        return masked

    def _analyze_contour_for_gel(self,
                                 contour: np.ndarray,
                                 height: int,
                                 width: int) -> Optional[GelLine]:
        """Analyze a contour (wave-like curve) for gel characteristics."""
        # Get contour points
        points = contour.reshape(-1, 2)

        if len(points) < 10:
            return None

        # Calculate bounding box
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Calculate spans
        x_span = (x_max - x_min) / width
        y_span = (y_max - y_min) / height

        # Skip if horizontal span is too small (not spanning across vial)
        if x_span < self.min_x_span:
            return None

        # Skip if too vertical (we want mostly horizontal waves)
        if y_span > x_span * 0.5:  # If height > 50% of width, too vertical
            return None

        # Calculate average y position (center of wave)
        y_center = np.mean(y_coords) / height

        # Calculate sinuous from the contour shape
        sinuous = self._calculate_contour_sinuous(points, width, height)

        # Calculate discontinuity (how much the contour breaks)
        discontinuity = self._calculate_contour_discontinuity(points, x_span, width)

        # Calculate strength based on contour length vs straight line distance
        contour_length = cv2.arcLength(contour, False)
        straight_distance = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        strength = min(1.0, contour_length / (straight_distance * 1.5)) if straight_distance > 0 else 0

        return GelLine(
            y_position=y_center,
            x_start=x_min / width,
            x_end=x_max / width,
            sinuous=sinuous,
            discontinuity=discontinuity,
            strength=strength,
            length_pixels=int(contour_length)
        )

    def _calculate_contour_sinuous(self,
                                      points: np.ndarray,
                                      width: int,
                                      height: int) -> float:
        """
        Calculate sinuous of a contour (wave-like pattern).
        Measures how much the curve deviates from a straight line.
        """
        if len(points) < 10:
            return 0.0

        # Sort points by x-coordinate
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        x = sorted_points[:, 0]
        y = sorted_points[:, 1]

        # Fit a polynomial (degree 2 for gentle curves)
        if len(x) > 3:
            try:
                coeffs = np.polyfit(x, y, 2)
                fitted_y = np.polyval(coeffs, x)

                # Calculate deviation from fitted curve
                deviations = np.abs(y - fitted_y)
                avg_deviation = np.mean(deviations) / height

                # Also measure amplitude of the wave
                y_range = (np.max(y) - np.min(y)) / height

                # Combine both metrics
                sinuous = min(1.0, (avg_deviation * 20 + y_range * 5))

                return float(sinuous)
            except:
                pass

        # Fallback: simple range measure
        y_range = (np.max(y) - np.min(y)) / height
        return float(min(1.0, y_range * 5))

    def _calculate_contour_discontinuity(self,
                                         points: np.ndarray,
                                         x_span: float,
                                         width: int) -> float:
        """
        Calculate how discontinuous/broken the contour is.
        """
        if len(points) < 5:
            return 0.0

        # Sort by x
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        # Calculate gaps in x-direction
        x_coords = sorted_points[:, 0]
        x_gaps = np.diff(x_coords)

        # Identify significant gaps (normalized)
        threshold = width * 0.02  # 2% of width
        large_gaps = x_gaps[x_gaps > threshold]

        if len(x_gaps) == 0:
            return 0.0

        # Calculate discontinuity score
        gap_fraction = len(large_gaps) / len(x_gaps)

        # Also consider if the contour doesn't span continuously
        total_gaps = np.sum(large_gaps)
        expected_span = (x_coords[-1] - x_coords[0])
        gap_ratio = total_gaps / expected_span if expected_span > 0 else 0

        discontinuity = min(1.0, gap_fraction * 2 + gap_ratio)

        return float(discontinuity)

    def _create_summary(self,
                        all_lines: List[GelLine],
                        gel_lines: List[GelLine],
                        height: int,
                        width: int,
                        is_gelled: bool) -> Dict[str, Any]:
        """Create summary of gel detection results."""
        if not gel_lines:
            return {
                'state': 'not_gelled' if not is_gelled else 'gelled',
                'message': 'No gel-like lines detected',
                'avg_sinuous': 0.0,
                'avg_discontinuity': 0.0
            }

        # Calculate average properties of gel lines
        avg_sinuous = np.mean([l.sinuous for l in gel_lines])
        avg_discontinuity = np.mean([l.discontinuity for l in gel_lines])
        avg_span = np.mean([l.x_span for l in gel_lines])

        # Vertical distribution
        y_positions = [l.y_position for l in gel_lines]
        vertical_spread = max(y_positions) - min(y_positions)

        description = self._generate_description(
            len(gel_lines), avg_sinuous, avg_discontinuity,
            avg_span, vertical_spread, is_gelled
        )

        return {
            'state': 'gelled' if is_gelled else 'not_gelled',
            'avg_sinuous': float(avg_sinuous),
            'avg_discontinuity': float(avg_discontinuity),
            'avg_horizontal_span': float(avg_span),
            'vertical_spread': float(vertical_spread),
            'description': description
        }

    def _generate_description(self,
                              num_lines: int,
                              avg_sinuous: float,
                              avg_discontinuity: float,
                              avg_span: float,
                              vertical_spread: float,
                              is_gelled: bool) -> str:
        """Generate human-readable description."""
        state = "GELLED" if is_gelled else "NOT GELLED"

        desc = (f"{state}: Detected {num_lines} gel-like lines "
                f"(sinuous and discontinuous). "
                f"Average sinuous: {avg_sinuous:.3f}, "
                f"discontinuity: {avg_discontinuity:.3f}, "
                f"horizontal span: {avg_span:.3f}, "
                f"vertical spread: {vertical_spread:.3f}")

        return desc

    def _line_to_dict(self, line: GelLine) -> Dict[str, Any]:
        """Convert line object to dictionary."""
        return {
            'y_position': float(line.y_position),
            'x_start': float(line.x_start),
            'x_end': float(line.x_end),
            'x_span': float(line.x_span),
            'sinuous': float(line.sinuous),
            'discontinuity': float(line.discontinuity),
            'strength': float(line.strength),
            'length_pixels': int(line.length_pixels),
            'is_gel_like': bool(line.is_gel_like)
        }

    def visualize(self,
                  image_path: Path,
                  output_path: Path,
                  lines: Optional[List[Dict[str, Any]]] = None,
                  show_all: bool = False) -> Path:
        """
        Create visualization of detected gel lines.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            lines: Optional pre-detected lines
            show_all: Show all lines or only gel-like lines

        Returns:
            Path to saved visualization
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        H, W = img.shape[:2]

        # Detect lines if not provided
        if lines is None:
            result = self.detect(image_path)
            all_lines = result['lines']
            gel_lines = result['gel_like_lines']
            is_gelled = result['is_gelled']
        else:
            all_lines = lines
            gel_lines = [l for l in lines if l['is_gel_like']]
            is_gelled = len(gel_lines) >= self.min_gel_lines

        # Create overlay
        overlay = img.copy()

        # Draw exclusion zones
        H, W = img.shape[:2]
        top_line = int(H * self.top_exclusion)
        bottom_line = int(H * (1 - self.bottom_exclusion))
        cv2.line(overlay, (0, top_line), (W, top_line), (0, 0, 255), 1)
        cv2.line(overlay, (0, bottom_line), (W, bottom_line), (0, 0, 255), 1)
        cv2.putText(overlay, "Exclusion", (5, top_line - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(overlay, "Exclusion", (5, bottom_line + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw lines
        lines_to_draw = gel_lines if not show_all else all_lines

        for i, line_dict in enumerate(lines_to_draw):
            y = int(line_dict['y_position'] * H)
            x1 = int(line_dict['x_start'] * W)
            x2 = int(line_dict['x_end'] * W)

            # Color based on gel characteristics
            is_gel = line_dict['is_gel_like']
            color = (0, 255, 0) if is_gel else (128, 128, 128)
            thickness = 3 if is_gel else 1

            # Draw wavy line representation
            cv2.line(overlay, (x1, y), (x2, y), color, thickness)

            # Draw bounding box to show wave extent
            if is_gel:
                # Approximate wave height as ±5% of image height
                wave_height = int(H * 0.05)
                cv2.rectangle(overlay,
                              (x1, y - wave_height),
                              (x2, y + wave_height),
                              color, 1)

            # Draw label for gel-like lines
            if is_gel:
                label = f"Wave{i + 1}"
                sinuous = line_dict['sinuous']
                cv2.putText(overlay, label, (x2 + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(overlay, f"w:{sinuous:.2f}", (x2 + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Blend with original
        alpha = 0.6
        output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Add text summary
        state_text = "STATE: GELLED" if is_gelled else "STATE: NOT GELLED"
        color = (0, 255, 0) if is_gelled else (0, 0, 255)

        cv2.putText(output, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(output, f"Gel lines: {len(gel_lines)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output)

        return output_path


def main():
    """Main function for standalone testing."""
    parser = argparse.ArgumentParser(
        description="Detect gel-like lines in vial images"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save visualization (default: <image>_gel_detection.png)")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save intermediate processing steps")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all lines, not just gel-like ones")

    # Detection parameters
    parser.add_argument("--min-gel-lines", type=int, default=3,
                        help="Minimum gel-like lines for gelled state")
    parser.add_argument("--min-sinuous", type=float, default=0.15,
                        help="Minimum sinuous threshold")
    parser.add_argument("--min-discontinuity", type=float, default=0.1,
                        help="Minimum discontinuity threshold")
    parser.add_argument("--min-x-span", type=float, default=0.2,
                        help="Minimum horizontal span")

    # Exclusion zones
    parser.add_argument("--top-exclusion", type=float, default=0.15,
                        help="Exclude top N%% of image (0-1, default 0.15 = 15%%)")
    parser.add_argument("--bottom-exclusion", type=float, default=0.15,
                        help="Exclude bottom N%% of image (0-1, default 0.15 = 15%%)")

    # Output options
    parser.add_argument("--json", type=str, default=None,
                        help="Save results as JSON to this file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")

    args = parser.parse_args()

    # Setup paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_gel_detection.png"

    # Initialize detector with custom parameters
    detector = GelLineDetector(
        min_gel_lines=args.min_gel_lines,
        min_sinuous=args.min_sinuous,
        min_discontinuity=args.min_discontinuity,
        min_x_span=args.min_x_span,
        top_exclusion=args.top_exclusion,
        bottom_exclusion=args.bottom_exclusion
    )

    print(f"Processing: {image_path}")
    print(f"Parameters: min_gel_lines={args.min_gel_lines}, "
          f"min_sinuous={args.min_sinuous}, "
          f"min_discontinuity={args.min_discontinuity}")
    print(f"Exclusion zones: top={args.top_exclusion * 100:.0f}%, bottom={args.bottom_exclusion * 100:.0f}%")
    print()

    # Run detection
    result = detector.detect(image_path, save_intermediates=args.save_intermediates)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return 1

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"State: {'GELLED' if result['is_gelled'] else 'NOT GELLED'}")
    print(f"Gel-like lines detected: {result['num_gel_lines']} / {result['total_lines']} total")
    print()
    print(result['summary']['description'])
    print()

    if args.verbose:
        print("Detailed line information:")
        print("-" * 60)
        for i, line in enumerate(result['gel_like_lines']):
            print(f"Line {i + 1}:")
            print(f"  Position: y={line['y_position']:.3f}, "
                  f"x=[{line['x_start']:.3f}, {line['x_end']:.3f}]")
            print(f"  sinuous: {line['sinuous']:.3f}")
            print(f"  Discontinuity: {line['discontinuity']:.3f}")
            print(f"  Horizontal span: {line['x_span']:.3f}")
            print()

    # Create visualization
    print(f"Saving visualization to: {output_path}")
    detector.visualize(image_path, output_path, show_all=args.show_all)

    # Save JSON if requested
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON results to: {json_path}")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())