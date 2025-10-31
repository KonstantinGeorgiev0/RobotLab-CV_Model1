"""
line_hv_detection.py
Detects both horizontal and vertical lines, with horizontal lines constrained to vial boundaries.
"""

import argparse
import json

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypeVar, Tuple
from dataclasses import dataclass
import sys

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import LINE_PARAMS


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
        return (self.x_start + self.x_end) / 2.0

    @property
    def x_length_frac(self) -> float:
        return self.x_end - self.x_start

    def __repr__(self) -> str:
        return (
            f"HorizontalLine(y={self.y_position:.3f}, "
            f"x=[{self.x_start:.3f}, {self.x_end:.3f}], "
            f"strength={self.strength:.3f})"
                )


@dataclass
class VerticalLine:
    """Represents a detected vertical line."""
    x_position: float  # Normalized x position (0=left, 1=right)
    y_start: float  # Normalized y start (0=top, 1=bottom)
    y_end: float  # Normalized y end
    thickness: int  # Line thickness in pixels
    strength: float  # Line strength/confidence (0-1)

    @property
    def y_center(self) -> float:
        return (self.y_start + self.y_end) / 2.0

    @property
    def y_length_frac(self) -> float:
        return self.y_end - self.y_start

    def __repr__(self) -> str:
        return (f"VerticalLine(x={self.x_position:.3f}, "
                f"y=[{self.y_start:.3f}, {self.y_end:.3f}], "
                f"strength={self.strength:.3f})"
                )


# define typevar T for type hinting
T = TypeVar('T', HorizontalLine, VerticalLine)


class LineDetector:
    """Detector for horizontal and vertical lines in vial images."""

    def __init__(self,
                 horiz_kernel_div: int = LINE_PARAMS["horiz_kernel_div"],
                 vert_kernel_div: int = LINE_PARAMS["vert_kernel_div"],
                 adaptive_block: int = LINE_PARAMS["adaptive_block"],
                 adaptive_c: int = LINE_PARAMS["adaptive_c"],
                 min_line_length: float = LINE_PARAMS["horiz_kernel_div"],
                 min_line_strength: float = LINE_PARAMS["min_line_length"],
                 merge_threshold: float = LINE_PARAMS["merge_threshold"]):
        """Initialize detector with parameters."""
        self.horiz_kernel_div = horiz_kernel_div
        self.vert_kernel_div = vert_kernel_div
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.min_line_length = min_line_length
        self.min_line_strength = min_line_strength
        self.merge_threshold = merge_threshold


    def detect(self,
               image_path: Path,
               top_exclusion: float = LINE_PARAMS["top_exclusion"],
               bottom_exclusion: float = LINE_PARAMS["bottom_exclusion"],
               save_debug: bool = False,
               debug_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Detect both horizontal and vertical lines.

        Args:
            image_path: Path to input image
            top_exclusion: Fraction of top to exclude (0-1)
            bottom_exclusion: Fraction of bottom to exclude (0-1)
            save_debug: Whether to save debug images
            debug_dir: Directory for debug images

        Returns:
            Dictionary with detection results
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': 'Failed to load image'}

        H, W = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # Prepare binary image
        inv = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(
            inv, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block,
            self.adaptive_c
        )

        # Extract horizontal lines
        horizontal_img = self._extract_lines(bw, W, 'horizontal')

        # Extract vertical lines
        vertical_img = self._extract_lines(bw, H, 'vertical')

        # Save debug images if requested
        if save_debug and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            cv2.imwrite(str(debug_dir / f"{stem}_gray.png"), gray)
            cv2.imwrite(str(debug_dir / f"{stem}_binary.png"), bw)
            cv2.imwrite(str(debug_dir / f"{stem}_horizontal.png"), horizontal_img)
            cv2.imwrite(str(debug_dir / f"{stem}_vertical.png"), vertical_img)

        # Analyze vertical lines to find vial boundaries
        vertical_lines = self._analyze_vertical_lines(vertical_img, H, W)
        vertical_lines = self._merge_nearby_vertical_lines(vertical_lines)
        vertical_lines = [l for l in vertical_lines
                         if l.y_length_frac >= self.min_line_length
                         and l.strength >= self.min_line_strength]
        vertical_lines.sort(key=lambda l: l.x_position)

        # Find leftmost and rightmost vertical lines (vial edges)
        left_boundary = vertical_lines[0].x_position if vertical_lines else 0.0
        right_boundary = vertical_lines[-1].x_position if vertical_lines else 1.0

        # Analyze horizontal lines within vial boundaries
        horizontal_lines = self._analyze_horizontal_lines(
            horizontal_img, H, W,
            top_exclusion, bottom_exclusion,
            left_boundary, right_boundary
        )
        horizontal_lines = self._merge_nearby_horizontal_lines(horizontal_lines)
        horizontal_lines = [l for l in horizontal_lines
                           if l.x_length_frac >= self.min_line_length
                           and l.strength >= self.min_line_strength]
        horizontal_lines.sort(key=lambda l: l.y_position)

        # Create summaries
        h_summary = self._create_line_summary(horizontal_lines, H, W)
        v_summary = self._create_line_summary(vertical_lines, H, W)

        return {
            'horizontal_lines': {
                'num_lines': len(horizontal_lines),
                'lines': [self._line_to_dic(l) for l in horizontal_lines],
                'summary': h_summary
            },
            'vertical_lines': {
                'num_lines': len(vertical_lines),
                'lines': [self._line_to_dic(l) for l in vertical_lines],
                'summary': v_summary,
                'left_boundary': float(left_boundary),
                'right_boundary': float(right_boundary)
            },
            'image_size': {'height': H, 'width': W}
        }


    def _extract_lines(self, bw: np.ndarray, size: int, orientation: str
                       ) -> np.ndarray:
        """Extract lines using morphology."""
        line = np.copy(bw)

        if orientation == 'horizontal':
            line_size = int(np.clip(size / self.horiz_kernel_div, 5, 51))
            line_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (line_size, 1)
            )
        elif orientation =='vertical':
            line_size = int(np.clip(size / self.vert_kernel_div, 5, 51))
            line_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, line_size)
            )
        else:
            raise ValueError("Invalid orientation. Use 'horizontal' or 'vertical'.")

        line = cv2.erode(line, line_kernel)
        line = cv2.dilate(line, line_kernel)
        return line


    def _analyze_lines(self,
                       line_img: np.ndarray,
                       height: int,
                       width: int,
                       line_type: str,
                       top_exclusion: float = 0.0,
                       bottom_exclusion: float = 0.0,
                       left_boundary: float = 0.0,
                       right_boundary: float = 1.0) -> List[Union[HorizontalLine, VerticalLine]]:
        """Generic method to analyze lines in an image."""
        lines = []

        if line_type == 'horizontal':
            # Calculate exclusion zones
            top_idx = int(height * top_exclusion)
            bottom_idx = int(height * (1 - bottom_exclusion))
            top_idx = max(0, min(top_idx, height))
            bottom_idx = max(0, min(bottom_idx, height))

            if bottom_idx <= top_idx:
                return lines

            # Convert boundary to pixel coordinates
            left_pixel = int(left_boundary * width)
            right_pixel = int(right_boundary * width)

            for y in range(top_idx, bottom_idx):
                # Only look at pixels within vial boundaries
                row = line_img[y, left_pixel:right_pixel]

                white_pixels = np.where(row > 127)[0]
                if len(white_pixels) == 0:
                    continue

                # Group continuous segments
                segments = self._group_segments(white_pixels)

                # Create line objects for significant segments
                for segment in segments:
                    if len(segment) < (right_pixel - left_pixel) * 0.1:
                        continue

                    # Adjust x positions relative to full image
                    x_start = (left_pixel + min(segment)) / width
                    x_end = (left_pixel + max(segment)) / width
                    y_norm = y / height

                    strength = len(segment) / (max(segment) - min(segment) + 1)

                    line = HorizontalLine(
                        y_position=y_norm,
                        x_start=x_start,
                        x_end=x_end,
                        thickness=1,
                        strength=strength
                    )
                    lines.append(line)

        elif line_type == 'vertical':
            for x in range(width):
                col = line_img[:, x]

                white_pixels = np.where(col > 127)[0]
                if len(white_pixels) == 0:
                    continue

                # Group continuous segments
                segments = self._group_segments(white_pixels)

                # Create line objects for significant segments
                for segment in segments:
                    if len(segment) < height * 0.1:
                        continue

                    y_start = min(segment) / height
                    y_end = max(segment) / height
                    x_norm = x / width

                    strength = len(segment) / (max(segment) - min(segment) + 1)

                    line = VerticalLine(
                        x_position=x_norm,
                        y_start=y_start,
                        y_end=y_end,
                        thickness=1,
                        strength=strength
                    )
                    lines.append(line)
        else:
            raise ValueError(f"Invalid line_type: {line_type}. Use 'horizontal' or 'vertical'.")

        return lines

    def _analyze_horizontal_lines(self,
                                  horizontal_img: np.ndarray,
                                  height: int,
                                  width: int,
                                  top_exclusion: float,
                                  bottom_exclusion: float,
                                  left_boundary: float,
                                  right_boundary: float) -> List[HorizontalLine]:
        """Analyze horizontal lines within vial boundaries."""
        return self._analyze_lines(
            horizontal_img, height, width, 'horizontal',
            top_exclusion, bottom_exclusion, left_boundary, right_boundary
        )

    def _analyze_vertical_lines(self,
                                vertical_img: np.ndarray,
                                height: int,
                                width: int) -> List[VerticalLine]:
        """Analyze vertical lines."""
        return self._analyze_lines(vertical_img, height, width, 'vertical')

    def _group_segments(self, pixels: np.ndarray, max_gap: int = 2) -> List[List[int]]:
        """Group continuous pixel segments allowing small gaps."""
        if len(pixels) == 0:
            return []

        segments = []
        current_segment = [pixels[0]]

        for i in range(1, len(pixels)):
            if pixels[i] - pixels[i - 1] <= max_gap:
                current_segment.append(pixels[i])
            else:
                if len(current_segment) > 0:
                    segments.append(current_segment)
                current_segment = [pixels[i]]

        if len(current_segment) > 0:
            segments.append(current_segment)

        return segments


    def _merge_nearby_lines(self, lines: List[T]) -> List[T]:
        """Generic method to merge lines that are close to each other."""
        if len(lines) <= 1:
            return lines

        # Determine sorting key and position comparison based on line type
        if not lines:
            return lines

        if isinstance(lines[0], HorizontalLine):
            sort_key = lambda l: l.y_position
            position_key = lambda l: l.y_position
        elif isinstance(lines[0], VerticalLine):
            sort_key = lambda l: l.x_position
            position_key = lambda l: l.x_position
        else:
            raise ValueError(f"Unsupported line type: {type(lines[0])}")

        lines = sorted(lines, key=sort_key)
        merged = []
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            line = lines[i]
            prev_line = current_group[-1]

            if abs(position_key(line) - position_key(prev_line)) < self.merge_threshold:
                current_group.append(line)
            else:
                merged.append(self._merge_line_group(current_group))
                current_group = [line]

        if current_group:
            merged.append(self._merge_line_group(current_group))

        return merged


    def _merge_nearby_horizontal_lines(self, lines: List[HorizontalLine]) -> List[HorizontalLine]:
        """Merge horizontal lines that are close vertically."""
        return self._merge_nearby_lines(lines)


    def _merge_nearby_vertical_lines(self, lines: List[VerticalLine]) -> List[VerticalLine]:
        """Merge vertical lines that are close horizontally."""
        return self._merge_nearby_lines(lines)


    def _merge_line_group(self, group: List[T]) -> T:
        """Method to merge a group of lines (horizontal or vertical)."""
        if len(group) == 1:
            return group[0]

        total_strength = sum(l.strength for l in group)
        strength_avg = total_strength / len(group)
        thickness = sum(l.thickness for l in group)

        if isinstance(group[0], HorizontalLine):
            # Merge horizontal lines
            y_avg = sum(l.y_position * l.strength for l in group) / total_strength
            x_start = min(l.x_start for l in group)
            x_end = max(l.x_end for l in group)

            return HorizontalLine(
                y_position=y_avg,
                x_start=x_start,
                x_end=x_end,
                thickness=thickness,
                strength=strength_avg
            )

        elif isinstance(group[0], VerticalLine):
            # Merge vertical lines
            x_avg = sum(l.x_position * l.strength for l in group) / total_strength
            y_start = min(l.y_start for l in group)
            y_end = max(l.y_end for l in group)

            return VerticalLine(
                x_position=x_avg,
                y_start=y_start,
                y_end=y_end,
                thickness=thickness,
                strength=strength_avg
            )

        else:
            raise ValueError(f"Unsupported line type: {type(group[0])}")


    def _merge_horizontal_group(self, group: List[HorizontalLine]) -> HorizontalLine:
        """Merge a group of horizontal lines."""
        return self._merge_line_group(group)


    def _merge_vertical_group(self, group: List[VerticalLine]) -> VerticalLine:
        """Merge a group of vertical lines."""
        return self._merge_line_group(group)


    def _create_line_summary(self, lines: List[Union[HorizontalLine, VerticalLine]],
                             height: int, width: int) -> Dict[str, Any]:
        """Create summary for lines."""
        if not lines:
            return {'message': 'No horizontal lines detected'}

        if isinstance(lines[0], HorizontalLine):
            y_positions = [l.y_position for l in lines]
            return {
                'vertical_span': {
                    'min': float(min(y_positions)),
                    'max': float(max(y_positions)),
                    'range': float(max(y_positions) - min(y_positions))
                },
                'average_line_length': float(np.mean([l.x_length_frac for l in lines])),
                'average_line_strength': float(np.mean([l.strength for l in lines]))
            }

        else:
            x_positions = [l.x_position for l in lines]
            return {
                'horizontal_span': {
                    'min': float(min(x_positions)),
                    'max': float(max(x_positions)),
                    'range': float(max(x_positions) - min(x_positions))
                },
                'average_line_length': float(np.mean([l.y_length_frac for l in lines])),
                'average_line_strength': float(np.mean([l.strength for l in lines]))
            }


    def _line_to_dic(self, line: Union[HorizontalLine, VerticalLine]) -> Dict[str, Any]:
        """Convert line to dictionary."""

        # if line is horizontal
        if isinstance(line, HorizontalLine):
            return {
                'type': type(line).__name__,
                'y_position': float(line.y_position),
                'x_start': float(line.x_start),
                'x_end': float(line.x_end),
                'x_center': float(line.x_center),
                'x_length_frac': float(line.x_length_frac),
                'thickness': int(line.thickness),
                'strength': float(line.strength)
            }
        # if line is vertical
        else:
            return {
                'type': type(line).__name__,
                'x_position': float(line.x_position),
                'y_start': float(line.y_start),
                'y_end': float(line.y_end),
                'y_center': float(line.y_center),
                'y_length_frac': float(line.y_length_frac),
                'thickness': int(line.thickness),
                'strength': float(line.strength)
            }


    def visualize(self,
                  image_path: Path,
                  output_path: Path,
                  result: Optional[Dict[str, Any]] = None,
                  top_exclusion: float = 0.0,
                  bottom_exclusion: float = 0.0) -> Path:
        """Create visualization of detected lines."""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        H, W = img.shape[:2]

        # Detect lines if not provided
        if result is None:
            result = self.detect(image_path, top_exclusion, bottom_exclusion)

        overlay = img.copy()

        # Draw exclusion zones
        if top_exclusion > 0:
            top_line_y = int(top_exclusion * H)
            cv2.line(overlay, (0, top_line_y), (W, top_line_y), (255, 0, 0), 2)
        if bottom_exclusion > 0:
            bottom_line_y = int((1 - bottom_exclusion) * H)
            cv2.line(overlay, (0, bottom_line_y), (W, bottom_line_y), (255, 0, 0), 2)

        # Draw vertical lines (red)
        for i, line_dict in enumerate(result['vertical_lines']['lines']):
            x = int(line_dict['x_position'] * W)
            y1 = int(line_dict['y_start'] * H)
            y2 = int(line_dict['y_end'] * H)

            cv2.line(overlay, (x, y1), (x, y2), (0, 0, 255), 2)
            label = f"V{i + 1}"
            cv2.putText(overlay, label, (x + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw vial boundaries (purple dashed)
        if result['vertical_lines']['num_lines'] > 0:
            left_x = int(result['vertical_lines']['left_boundary'] * W)
            right_x = int(result['vertical_lines']['right_boundary'] * W)

            for y in range(0, H, 20):
                cv2.line(overlay, (left_x, y), (left_x, min(y+10, H)), (255, 0, 255), 2)
                cv2.line(overlay, (right_x, y), (right_x, min(y+10, H)), (255, 0, 255), 2)

        # Draw horizontal lines (green)
        for i, line_dict in enumerate(result['horizontal_lines']['lines']):
            y = int(line_dict['y_position'] * H)
            x1 = int(line_dict['x_start'] * W)
            x2 = int(line_dict['x_end'] * W)

            cv2.line(overlay, (x1, y), (x2, y), (0, 255, 0), 2)
            label = f"H{i + 1}: y={line_dict['y_position']:.3f}"
            cv2.putText(overlay, label, (x2 + 5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Blend
        alpha = 0.6
        output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Add text summary
        h_lines = result['horizontal_lines']['num_lines']
        v_lines = result['vertical_lines']['num_lines']
        summary = f"H: {h_lines}, V: {v_lines}"
        cv2.putText(output, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        cv2.imwrite(str(output_path), output)
        return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Detect horizontal and vertical lines in vial images"
    )
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    ap.add_argument("-o", "--outdir", default="line_hv_detection_results",
                   help="Output directory")
    ap.add_argument("--top-exclusion", type=float, default=LINE_PARAMS.get("top_exclusion", 0.30),
                   help="Fraction of top to exclude (0-1)")
    ap.add_argument("--bottom-exclusion", type=float, default=LINE_PARAMS.get("bottom_exclusion", 0.15),
                   help="Fraction of bottom to exclude (0-1)")
    ap.add_argument("--min-line-length", type=float, default=LINE_PARAMS.get("min_line_length", 0.75),
                   help="Minimum line length as fraction")
    ap.add_argument("--merge-threshold", type=float, default=LINE_PARAMS.get("merge_threshold", 0.1),
                   help="Threshold for merging nearby lines")
    ap.add_argument("--save-debug", action="store_true",
                   help="Save debug images")

    args = ap.parse_args()
    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    # Initialize detector
    detector = LineDetector(
        horiz_kernel_div=LINE_PARAMS.get("horiz_kernel_div", 15),
        vert_kernel_div=LINE_PARAMS.get("vert_kernel_div", 30),
        adaptive_c=LINE_PARAMS.get("adaptive_c", -2),
        adaptive_block=LINE_PARAMS.get("adaptive_block", 15),
        min_line_strength=LINE_PARAMS.get("min_line_strength", 0.8),
        min_line_length=args.min_line_length,
        merge_threshold=args.merge_threshold
    )

    # Detect lines
    debug_dir = outdir / "debug" if args.save_debug else None
    result = detector.detect(
        image_path,
        top_exclusion=args.top_exclusion,
        bottom_exclusion=args.bottom_exclusion,
        save_debug=args.save_debug,
        debug_dir=debug_dir
    )

    # Print results
    print("\n" + "="*60)
    print("VERTICAL LINES (Vial Boundaries)")
    print("="*60)
    print(f"Detected {result['vertical_lines']['num_lines']} vertical lines")
    if result['vertical_lines']['num_lines'] > 0:
        print(f"Left boundary:  x={result['vertical_lines']['left_boundary']:.3f}")
        print(f"Right boundary: x={result['vertical_lines']['right_boundary']:.3f}")
        print("\nDetailed vertical line information:")
        for i, line in enumerate(result['vertical_lines']['lines']):
            print(f"  V{i+1}: x={line['x_position']:.3f}, "
                  f"y=[{line['y_start']:.3f}, {line['y_end']:.3f}], "
                  f"line_length={line['y_length_frac']:.3f}")

    print("\n" + "="*60)
    print("HORIZONTAL LINES (Phase Boundaries)")
    print("="*60)
    print(f"Detected {result['horizontal_lines']['num_lines']} horizontal lines")
    if result['horizontal_lines']['num_lines'] > 0:
        print("\nDetailed horizontal line information:")
        for i, line in enumerate(result['horizontal_lines']['lines']):
            print(f"  H{i+1}: y={line['y_position']:.3f}, "
                  f"x=[{line['x_start']:.3f}, {line['x_end']:.3f}], "
                  f"line_length={line['x_length_frac']:.3f}")

    # Create summary json file
    summary_path = outdir / f"{image_path.stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Create visualization
    viz_path = outdir / f"{image_path.stem}_lines.png"
    detector.visualize(
        image_path, viz_path, result,
        args.top_exclusion, args.bottom_exclusion
    )

    print(f"\nVisualization saved to: {viz_path}")
    if args.save_debug:
        print(f"Debug images saved to: {debug_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
