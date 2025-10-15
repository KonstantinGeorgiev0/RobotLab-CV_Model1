"""
Horizontal line detection module.
Detects horizontal lines in vial images and provides spatial analysis.
"""
import argparse

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import sys


@dataclass
class HorizontalLine:
    """Represents a detected horizontal line."""
    y_position: float  # Normalized y position (0=top, 1=bottom)
    x_start: float  # Normalized x start (0=left, 1=right)
    x_end: float  # Normalized x end
    thickness: int  # Line thickness in pixels
    strength: float  # Line strength/confidence (0-1)

    @property
    def x_center(self) -> float:
        """Center x position."""
        return (self.x_start + self.x_end) / 2.0

    @property
    def x_span(self) -> float:
        """Horizontal span of line."""
        return self.x_end - self.x_start

    def __repr__(self) -> str:
        return (f"HorizontalLine(y={self.y_position:.3f}, "
                f"x=[{self.x_start:.3f}, {self.x_end:.3f}], "
                f"strength={self.strength:.3f})")


class HorizontalLineDetector:
    """Detector for horizontal lines in vial images."""

    def __init__(self,
                 horiz_kernel_div: int = 15,
                 vert_kernel_div: int = 30,
                 adaptive_block: int = 15,
                 adaptive_c: int = 2,
                 min_line_length: float = 0.3,
                 min_line_strength: float = 0.1,
                 merge_threshold: float = 0.02):
        """
        Initialize horizontal line detector.

        Args:
            horiz_kernel_div: Divisor for horizontal kernel size
            vert_kernel_div: Divisor for vertical kernel size
            adaptive_block: Block size for adaptive threshold
            adaptive_c: Constant for adaptive threshold
            min_line_length: Minimum line length as fraction of image width
            min_line_strength: Minimum line strength (0-1)
            merge_threshold: Threshold for merging nearby lines (normalized)
        """
        self.horiz_kernel_div = horiz_kernel_div
        self.vert_kernel_div = vert_kernel_div
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.min_line_length = min_line_length
        self.min_line_strength = min_line_strength
        self.merge_threshold = merge_threshold

    def detect(self, image_path: Path, top_exclusion: float = 0.0, bottom_exclusion: float = 1.0) -> Dict[str, Any]:
        """
        Detect horizontal lines in image.

        Args:
            image_path: Path to input image
            top_exclusion: Fraction of top of image to exclude (0-1)
            bottom_exclusion: Fraction of bottom of image to exclude (0-1)

        Returns:
            Dictionary with detection results
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': 'Failed to load image'}

        H, W = img.shape[:2]

        # Extract horizontal lines
        horizontal_img = self._extract_horizontal_lines(img)

        # Analyze lines
        lines = self._analyze_lines(horizontal_img, H, W, top_exclusion, bottom_exclusion)

        # Merge nearby lines
        lines = self._merge_nearby_lines(lines)

        # Filter by minimum length and strength
        lines = [l for l in lines
                 if l.x_span >= self.min_line_length
                 and l.strength >= self.min_line_strength]

        # Sort by y position (top to bottom)
        lines.sort(key=lambda l: l.y_position)

        # Create summary
        summary = self._create_summary(lines, H, W)

        return {
            'num_lines': len(lines),
            'lines': [self._line_to_dict(l) for l in lines],
            'summary': summary,
            'image_size': {'height': H, 'width': W}
        }

    def _extract_horizontal_lines(self, img: np.ndarray) -> np.ndarray:
        """Extract horizontal lines using gradient analysis."""
        # Convert to grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Compute vertical gradient (horizontal edges)
        sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobelY = np.abs(sobelY)
        
        # Normalize to 0-255
        sobelY = np.uint8(255 * sobelY / np.max(sobelY))
        
        # Threshold to get strong horizontal edges
        _, horizontal = cv2.threshold(sobelY, 30, 255, cv2.THRESH_BINARY)
        
        # Enhance horizontal structures
        cols = horizontal.shape[1]
        horizontal_size = max(3, cols // 10)  # Smaller divisor for larger kernel
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_size, 1)
        )
        
        # Dilate to connect horizontal edges
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=2)
        
        # Debug
        cv2.imwrite('debug_gradient.png', sobelY)
        cv2.imwrite('debug_horizontal.png', horizontal)
        
        return horizontal

    # def _extract_horizontal_lines(self, img: np.ndarray) -> np.ndarray:
    #     """Extract horizontal lines using edge detection."""
    #     # Convert to grayscale
    #     if img.ndim == 3:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = img.copy()
        
    #     # Apply Gaussian blur to reduce noise
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
    #     # Detect edges using Canny
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     # Create horizontal kernel to enhance horizontal edges
    #     cols = edges.shape[1]
    #     horizontal_size = max(3, cols // self.horiz_kernel_div)
    #     horizontal_kernel = cv2.getStructuringElement(
    #         cv2.MORPH_RECT,
    #         (horizontal_size, 1)
    #     )
        
    #     # Dilate to connect nearby horizontal edges
    #     horizontal = cv2.dilate(edges, horizontal_kernel, iterations=1)
        
    #     # Optional: Close small gaps
    #     horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_kernel)
        
    #     # Debug output
    #     cv2.imwrite('debug_edges.png', edges)
    #     cv2.imwrite('debug_horizontal.png', horizontal)
        
    #     return horizontal
    
    # def _extract_horizontal_lines(self, img: np.ndarray) -> np.ndarray:
    #     """Extract horizontal lines using morphological operations."""
    #     # Convert to grayscale
    #     if img.ndim == 3:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = img.copy()

    #     # Invert and apply adaptive threshold
    #     inv = cv2.bitwise_not(gray)
    #     bw = cv2.adaptiveThreshold(
    #         inv, 255,
    #         cv2.ADAPTIVE_THRESH_MEAN_C,
    #         cv2.THRESH_BINARY,
    #         self.adaptive_block,
    #         self.adaptive_c
    #     )
        
    #     # Create horizontal kernel
    #     cols = bw.shape[1]
    #     horizontal_size = max(3, cols // self.horiz_kernel_div)
    #     print(f"Kernel size: {horizontal_size}x1 for image width {cols}")
    #     horizontal_kernel = cv2.getStructuringElement(
    #         cv2.MORPH_RECT,
    #         (horizontal_size, 1)
    #     )

    #     # Apply morphological operations
    #     horizontal = cv2.erode(bw, horizontal_kernel)
    #     horizontal = cv2.dilate(horizontal, horizontal_kernel)

    #     return horizontal

    def _analyze_lines(self,
                       horizontal_img: np.ndarray,
                       height: int,
                       width: int,
                       top_exclusion: float,
                       bottom_exclusion: float) -> List[HorizontalLine]:
        """Analyze horizontal image to extract individual lines, with optional exclusions."""
        lines = []
        top_idx = int(height * top_exclusion)  # start after top excluded
        bottom_idx = int(height * (1.0 - bottom_exclusion))  # stop before bottom excluded
        top_idx = max(0, min(top_idx, height))
        bottom_idx = max(0, min(bottom_idx, height))
        if bottom_idx <= top_idx:
            return lines  # nothing to scan

        # scan for each row
        for y in range(top_idx, bottom_idx):
            row = horizontal_img[y, :]

            # Find continuous white regions
            white_pixels = np.where(row > 127)[0]

            if len(white_pixels) == 0:
                continue

            # Group continuous segments
            segments = []
            current_segment = [white_pixels[0]]

            for i in range(1, len(white_pixels)):
                if white_pixels[i] - white_pixels[i - 1] <= 2:  # Allow small gaps
                    current_segment.append(white_pixels[i])
                else:
                    if len(current_segment) > 0:
                        segments.append(current_segment)
                    current_segment = [white_pixels[i]]

            if len(current_segment) > 0:
                segments.append(current_segment)

            # Create line objects for significant segments
            for segment in segments:
                if len(segment) < width * 0.1:  # Skip very short segments
                    continue

                x_start = min(segment) / width
                x_end = max(segment) / width
                y_norm = y / height

                # Calculate line strength (density of white pixels)
                strength = len(segment) / (max(segment) - min(segment) + 1)

                line = HorizontalLine(
                    y_position=y_norm,
                    x_start=x_start,
                    x_end=x_end,
                    thickness=1,
                    strength=strength
                )
                lines.append(line)

        return lines

    def _merge_nearby_lines(self, lines: List[HorizontalLine]) -> List[HorizontalLine]:
        """Merge lines that are very close vertically."""
        if len(lines) <= 1:
            return lines

        # Sort by y position
        lines = sorted(lines, key=lambda l: l.y_position)

        merged = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            line = lines[i]
            prev_line = current_group[-1]

            # Check if lines are close enough to merge
            if abs(line.y_position - prev_line.y_position) < self.merge_threshold:
                current_group.append(line)
            else:
                # Merge current group and start new one
                merged.append(self._merge_line_group(current_group))
                current_group = [line]

        # Merge last group
        if current_group:
            merged.append(self._merge_line_group(current_group))

        return merged

    def _merge_line_group(self, group: List[HorizontalLine]) -> HorizontalLine:
        """Merge a group of nearby lines into one."""
        if len(group) == 1:
            return group[0]

        # Average y position weighted by strength
        total_strength = sum(l.strength for l in group)
        y_avg = sum(l.y_position * l.strength for l in group) / total_strength

        # Take min/max x positions
        x_start = min(l.x_start for l in group)
        x_end = max(l.x_end for l in group)

        # Average strength
        strength_avg = total_strength / len(group)

        # Sum thickness
        thickness = sum(l.thickness for l in group)

        return HorizontalLine(
            y_position=y_avg,
            x_start=x_start,
            x_end=x_end,
            thickness=thickness,
            strength=strength_avg
        )

    def _create_summary(self,
                        lines: List[HorizontalLine],
                        height: int,
                        width: int) -> Dict[str, Any]:
        """Create summary statistics for detected lines."""
        if not lines:
            return {
                'message': 'No horizontal lines detected',
                'vertical_span': None,
                'horizontal_span': None,
                'description': 'No horizontal lines detected'
            }

        # Calculate spans
        y_positions = [l.y_position for l in lines]
        vertical_span = {
            'min': float(min(y_positions)),
            'max': float(max(y_positions)),
            'range': float(max(y_positions) - min(y_positions))
        }

        # Calculate overall horizontal span
        x_min = min(l.x_start for l in lines)
        x_max = max(l.x_end for l in lines)
        horizontal_span = {
            'min': float(x_min),
            'max': float(x_max),
            'range': float(x_max - x_min)
        }

        # Average line properties
        avg_length = np.mean([l.x_span for l in lines])
        avg_strength = np.mean([l.strength for l in lines])

        return {
            'vertical_span': vertical_span,
            'horizontal_span': horizontal_span,
            'average_line_length': float(avg_length),
            'average_line_strength': float(avg_strength),
            'description': self._generate_description(lines)
        }

    def _generate_description(self, lines: List[HorizontalLine]) -> str:
        """Generate human-readable description of lines."""
        if not lines:
            return "No horizontal lines detected"

        n = len(lines)
        y_positions = [l.y_position for l in lines]
        y_min, y_max = min(y_positions), max(y_positions)

        x_starts = [l.x_start for l in lines]
        x_ends = [l.x_end for l in lines]
        x_min, x_max = min(x_starts), max(x_ends)

        desc = (f"{n} horizontal line{'s' if n > 1 else ''} detected, "
                f"spanning between {y_min:.2f} and {y_max:.2f} vertically "
                f"(0=top, 1=bottom), "
                f"and between {x_min:.2f} and {x_max:.2f} horizontally "
                f"(0=left, 1=right)")

        return desc

    def _line_to_dict(self, line: HorizontalLine) -> Dict[str, Any]:
        """Convert line object to dictionary."""
        return {
            'y_position': float(line.y_position),
            'x_start': float(line.x_start),
            'x_end': float(line.x_end),
            'x_center': float(line.x_center),
            'x_span': float(line.x_span),
            'thickness': int(line.thickness),
            'strength': float(line.strength)
        }

    def visualize(self, image_path: Path, output_path: Path, 
                lines: Optional[List[Dict[str, Any]]] = None,
                top_exclusion: float = 0.0, bottom_exclusion: float = 1.0) -> Path:
        # Load image
        img = cv2.imread(str(image_path))
        H, W = img.shape[:2]
        
        # Detect lines if not provided
        if lines is None:
            result = self.detect(image_path, top_exclusion=top_exclusion,
                                bottom_exclusion=bottom_exclusion)
            lines = result['lines']
        
        overlay = img.copy()
        
        # DEBUG: Add exclusion zone visualization
        top_line_y = int(top_exclusion * H)
        bottom_line_y = int((1.0 - bottom_exclusion) * H)
        cv2.line(overlay, (0, top_line_y), (W, top_line_y), (255, 0, 0), 2)  # Blue for top
        cv2.line(overlay, (0, bottom_line_y), (W, bottom_line_y), (255, 0, 0), 2)  # Blue for bottom
        
        # Draw each line
        for i, line_dict in enumerate(lines):
            y = int(line_dict['y_position'] * H)  # This should be correct
            x1 = int(line_dict['x_start'] * W)
            x2 = int(line_dict['x_end'] * W)
            
            # DEBUG: Print line positions
            print(f"Drawing line {i+1}: y_norm={line_dict['y_position']:.3f}, "
                f"y_pixel={y}, x=[{x1}, {x2}]")
            
            # Draw line
            cv2.line(overlay, (x1, y), (x2, y), (0, 255, 0), 2)
            
            # Draw label
            label = f"L{i + 1}: y={line_dict['y_position']:.3f}"
            cv2.putText(overlay, label, (x2 + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
        # Blend with original
        alpha = 0.6
        output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Add text summary
        summary_text = f"Lines detected: {len(lines)}"
        cv2.putText(output, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        cv2.imwrite(str(output_path), output)
        return output_path


def integrate_with_classifier(
        crop_path: Path,
        label_path: Path,
        line_detector: HorizontalLineDetector) -> Dict[str, Any]:
    """
    Integration function for vial state classifier.

    Args:
        crop_path: Path to vial crop image
        label_path: Path to YOLO label file
        line_detector: HorizontalLineDetector instance

    Returns:
        Combined analysis with line detection
    """
    # Detect horizontal lines
    line_result = line_detector.detect(crop_path)

    # Determine implications for vial state
    num_lines = line_result['num_lines']

    if num_lines >= 2:
        # Multiple lines suggest phase separation
        state_hint = "phase_separated"
        confidence = "high" if num_lines >= 3 else "medium"
    elif num_lines == 1:
        # Single line might indicate stable or meniscus
        state_hint = "stable_or_gelled"
        confidence = "medium"
    else:
        # No lines detected
        state_hint = "gelled_or_air"
        confidence = "low"

    return {
        'line_detection': line_result,
        'num_lines': num_lines,
        'state_hint': state_hint,
        'confidence': confidence
    }


def debug_line_detection(image_path: Path, top_exclusion: float, bottom_exclusion: float):
    """Debug helper to understand line detection."""
    img = cv2.imread(str(image_path))
    H, W = img.shape[:2]
    
    print(f"\nImage dimensions: {W}x{H}")
    print(f"Exclusions: top={top_exclusion}, bottom={bottom_exclusion}")
    print(f"Scanning region: y={int(H*top_exclusion)} to y={int(H*(1-bottom_exclusion))}")
    print(f"Scanning {(1-top_exclusion-bottom_exclusion)*100:.1f}% of image height\n")
    
    detector = HorizontalLineDetector(
        min_line_length=0.3,
        merge_threshold=0.02
    )
    
    result = detector.detect(image_path, top_exclusion, bottom_exclusion)
    
    print(f"Lines detected: {result['num_lines']}")
    for i, line in enumerate(result['lines']):
        y_pixel = int(line['y_position'] * H)
        print(f"Line {i+1}: y_norm={line['y_position']:.3f}, "
              f"y_pixel={y_pixel}/{H}, "
              f"x_span={line['x_span']:.3f}")
    
    return result

def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--outdir", default="horizontal_lines_detection_results", help="Path to horizontal_line_detection_results image")
    ap.add_argument("--top-exclusion", type=float, default=0.2, help="Fraction of top of image to exclude (0-1)")
    ap.add_argument("--bottom-exclusion", type=float, default=0.2, help="Fraction of bottom of image to exclude (0-1)")
    ap.add_argument("--min-line-length", type=float, default=0.3, help="Minimum line length as fraction of image width")
    ap.add_argument("--merge-threshold", type=float, default=0.02, help="Threshold for merging nearby lines (normalized)")

    args = ap.parse_args()
    image_path = Path(args.image)
    outdir = Path(args.outdir)

    # Initialize detector
    detector = HorizontalLineDetector(
        min_line_length=args.min_line_length,
        min_line_strength=0.1,
        merge_threshold=args.merge_threshold
    )

    # Check if image exists
    if image_path.exists():
        # Detect lines
        result = detector.detect(image_path, top_exclusion=args.top_exclusion, bottom_exclusion=args.bottom_exclusion)

        # Print results
        print(f"Detected {result['num_lines']} horizontal lines:")
        print(result['summary']['description'])
        print("\nDetailed line information:")
        for i, line in enumerate(result['lines']):
            print(f"  Line {i + 1}: y={line['y_position']:.3f}, "
                  f"x=[{line['x_start']:.3f}, {line['x_end']:.3f}], "
                  f"strength={line['strength']:.3f}")

        # Create visualization
        viz_path = Path(outdir / f"{image_path.stem}_lines.jpg")
        viz_path.parent.mkdir(exist_ok=True)
        detector.visualize(image_path, viz_path, lines=result['lines'], top_exclusion=args.top_exclusion, bottom_exclusion=args.bottom_exclusion)

        print(f"\nVisualization saved to: {viz_path}")
        debug_line_detection(image_path, args.top_exclusion, args.bottom_exclusion)

    else:
        print(f"Image not found: {image_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
