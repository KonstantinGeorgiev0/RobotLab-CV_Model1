"""
Enhanced state classifier with integrated image analysis tools.
Combines detection-based classification with line detection and curve analysis.

Classification Hierarchy (in order of priority):
1. Only Air - No liquid content detected
2. Phase Separated - Multiple distinct liquid phases
3. Gelled - Gel-like characteristics detected
4. Stable - Normal liquid state
5. Unknown - Insufficient information
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from config import (
    CLASS_IDS, LIQUID_CLASSES, DETECTION_THRESHOLDS,
    ONLY_AIR_CLASSIFICATION, CURVE_PARAMS, LINE_PARAMS,
    REGION_EXCLUSION, PHASE_SEPARATION_THRESHOLDS
)
from robotlab_utils.io_utils import read_yolo_labels
from analysis.phase_separation import PhaseSeparationDetector
from analysis.gelled_analysis import run_curve_metrics
from image_analysis.line_hv_detection import LineDetector


@dataclass
class DetectionSummary:
    """Summary of YOLO detections with computed metrics."""
    all_detections: List[Dict[str, Any]]
    liquid_detections: List[Dict[str, Any]]
    air_detections: List[Dict[str, Any]]
    cap_detections: List[Dict[str, Any]]

    gel_count: int
    stable_count: int
    air_count: int
    cap_count: int

    total_liquid_area: float
    gel_area: float
    stable_area: float

    max_air_height_fraction: float
    max_vertical_gap: float

    height: int
    width: int

    @property
    def has_liquid(self) -> bool:
        return len(self.liquid_detections) > 0

    @property
    def has_air(self) -> bool:
        return len(self.air_detections) > 0

    @property
    def gel_area_fraction(self) -> float:
        return self.gel_area / self.total_liquid_area if self.total_liquid_area > 0 else 0.0

    @property
    def gel_count_fraction(self) -> float:
        total = self.gel_count + self.stable_count
        return self.gel_count / total if total > 0 else 0.0


class VialStateClassifier:
    """
    Classifier that integrates multiple analysis methods with clear priority.

    Analysis Methods:
    1. YOLO detection-based classification
    2. Horizontal line detection for phase separation
    3. Curved line analysis for gel detection
    4. Detection geometry analysis
    """

    def __init__(self,
                 use_line_detection: bool = True,
                 use_curved_line_detection: bool = True,
                 use_turbidity: bool = False,
                 merge_boxes: bool = True,
                 region_exclusion: Optional[Dict[str, float]] = None):
        """Initialize classifier with configuration."""
        self.use_line_detection = use_line_detection
        self.use_curved_line_detection = use_curved_line_detection
        self.use_turbidity = use_turbidity
        self.merge_boxes = merge_boxes
        self.region_exclusion = region_exclusion or REGION_EXCLUSION.copy()

        # Initialize sub-detectors
        self.phase_detector = PhaseSeparationDetector()
        self.line_detector = LineDetector(
            min_line_length=LINE_PARAMS['min_line_length'],
            merge_threshold=LINE_PARAMS['merge_threshold']
        ) if use_line_detection else None

    def classify(self,
                 crop_path: Path,
                 label_path: Path,
                 save_analysis_viz: bool = False,
                 viz_output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Classify vial state using all available methods.

        Returns:
            Dictionary with classification results and detailed metrics
        """
        # Load and validate image
        img = cv2.imread(str(crop_path))
        if img is None:
            return self._error_result("image_load_failed", crop_path)

        H, W = img.shape[:2]

        # Parse and process detections
        raw_detections = read_yolo_labels(label_path)
        detection_summary = self._create_detection_summary(raw_detections, H, W)

        # Run optional analysis modules
        line_analysis = self._run_line_analysis(
            crop_path, H, W, save_analysis_viz, viz_output_dir
        ) if self.use_line_detection and self.line_detector else None

        curve_analysis = self._run_curve_analysis(
            crop_path, save_analysis_viz, viz_output_dir
        ) if self.use_curved_line_detection else None

        # Perform hierarchical classification
        classification_result = self._hierarchical_classification(
            detection_summary, line_analysis, curve_analysis
        )

        # Compile comprehensive result
        return {
            **classification_result,
            "analysis_metrics": {
                "num_detections": len(raw_detections),
                "detection_summary": self._summary_to_dict(detection_summary),
                "line_detection": line_analysis,
                "curve_analysis": curve_analysis,
            }
        }

    def _create_detection_summary(self,
                                  detections: List[Dict[str, Any]],
                                  height: int,
                                  width: int) -> DetectionSummary:
        """
        Process raw YOLO detections into structured summary.
        Centralizes all detection processing to avoid redundant transformations.
        """
        all_dets = []
        liquid_dets = []
        air_dets = []
        cap_dets = []

        gel_count = stable_count = air_count = cap_count = 0
        gel_area = stable_area = total_liquid_area = 0.0
        max_air_height = 0.0

        for det in detections:
            # Convert to pixel coordinates
            cx = det['cx'] * width
            cy = det['cy'] * height
            w = det['w'] * width
            h = det['h'] * height

            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2

            processed = {
                'class_id': det['class_id'],
                'confidence': det.get('confidence', 1.0),
                'center_x': cx,
                'center_y': cy,
                'box': [x1, y1, x2, y2],
                'area': w * h,
                'width': w,
                'height': h
            }

            all_dets.append(processed)
            class_id = det['class_id']

            # Categorize by class
            if class_id == CLASS_IDS['GEL']:
                liquid_dets.append(processed)
                gel_count += 1
                gel_area += processed['area']
                total_liquid_area += processed['area']
            elif class_id == CLASS_IDS['STABLE']:
                liquid_dets.append(processed)
                stable_count += 1
                stable_area += processed['area']
                total_liquid_area += processed['area']
            elif class_id == CLASS_IDS['AIR']:
                air_dets.append(processed)
                air_count += 1
                max_air_height = max(max_air_height, h / height)
            elif class_id == CLASS_IDS['CAP']:
                cap_dets.append(processed)
                cap_count += 1

        # Calculate maximum vertical gap between liquid detections
        max_gap = self._calculate_max_vertical_gap(liquid_dets, height)

        return DetectionSummary(
            all_detections=all_dets,
            liquid_detections=liquid_dets,
            air_detections=air_dets,
            cap_detections=cap_dets,
            gel_count=gel_count,
            stable_count=stable_count,
            air_count=air_count,
            cap_count=cap_count,
            total_liquid_area=total_liquid_area,
            gel_area=gel_area,
            stable_area=stable_area,
            max_air_height_fraction=max_air_height,
            max_vertical_gap=max_gap,
            height=height,
            width=width
        )

    def _calculate_max_vertical_gap(self,
                                    liquid_dets: List[Dict[str, Any]],
                                    height: int) -> float:
        """Calculate maximum normalized vertical gap between liquid detections."""
        if len(liquid_dets) < 2:
            return 0.0

        sorted_dets = sorted(liquid_dets, key=lambda d: d['center_y'])
        max_gap = 0.0

        for i in range(len(sorted_dets) - 1):
            bottom = sorted_dets[i]['box'][3]
            top = sorted_dets[i + 1]['box'][1]
            gap = (top - bottom) / height
            max_gap = max(max_gap, gap)

        return max_gap

    def _run_line_analysis(self,
                           crop_path: Path,
                           height: int,
                           width: int,
                           save_viz: bool,
                           viz_dir: Optional[Path]) -> Dict[str, Any]:
        """Run horizontal and vertical line detection analysis."""
        try:
            result = self.line_detector.detect(
                image_path=crop_path,
                top_exclusion=self.region_exclusion.get('top_fraction', 0.0),
                bottom_exclusion=self.region_exclusion.get('bottom_fraction', 0.0),
                save_debug=False
            )

            # Save visualization if requested
            if save_viz and viz_dir:
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_path = viz_dir / f"{crop_path.stem}_line_detection.png"
                self.line_detector.visualize(
                    crop_path, viz_path, result,
                    self.region_exclusion.get('top_fraction', 0.0),
                    self.region_exclusion.get('bottom_fraction', 0.0)
                )

            # Extract metrics
            h_lines = result['horizontal_lines']['lines']

            return {
                'success': True,
                'num_horizontal_lines': len(h_lines),
                'horizontal_positions': [l['y_position'] for l in h_lines],
                'horizontal_lengths': [l['x_length_frac'] for l in h_lines],
                'num_vertical_lines': result['vertical_lines']['num_lines'],
                'vertical_boundaries': {
                    'left': result['vertical_lines'].get('left_boundary', 0.0),
                    'right': result['vertical_lines'].get('right_boundary', 1.0)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _run_curve_analysis(self,
                            crop_path: Path,
                            save_viz: bool,
                            viz_dir: Optional[Path]) -> Dict[str, Any]:
        """Run curve variance analysis for gel detection."""
        try:
            curve_result = run_curve_metrics(crop_path)

            # Save visualization if requested
            if save_viz and viz_dir and curve_result.get('stats'):
                from image_analysis.guided_curve_tracer import GuidedCurveTracer

                viz_dir.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(crop_path))

                if img is not None:
                    tracer = GuidedCurveTracer(
                        vertical_bounds=CURVE_PARAMS.get("vertical_bounds", (0.30, 0.80)),
                        horizontal_bounds=CURVE_PARAMS.get("horizontal_bounds", (0.05, 0.95)),
                        search_offset_frac=CURVE_PARAMS.get("search_offset_frac", 0.10),
                        median_kernel=CURVE_PARAMS.get("median_kernel", 9),
                        max_step_px=CURVE_PARAMS.get("max_step_px", 4)
                    )

                    xs, ys, metadata = tracer.trace_curve(img, crop_path, guide_y=None)

                    if len(xs) > 0:
                        viz_path = viz_dir / f"{crop_path.stem}_curve_analysis.png"
                        tracer.visualize(img, xs, ys, metadata, viz_path)

            stats = curve_result.get('stats', {})
            return {
                'success': True,
                'gelled_by_curve': curve_result.get('gelled_by_curve', False),
                'variance_from_baseline': stats.get('variance_from_baseline', 0.0),
                'std_dev_from_baseline': stats.get('std_dev_from_baseline', 0.0),
                'roughness': stats.get('roughness', 0.0),
                'num_points': stats.get('num_points', 0),
                'reason': curve_result.get('reason', '')
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _hierarchical_classification(self,
                                     det_summary: DetectionSummary,
                                     line_analysis: Optional[Dict[str, Any]],
                                     curve_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform classification using clear hierarchy.

        Priority Order:
        1. Only Air (no liquid content)
        2. Phase Separated (distinct phases detected)
        3. Gelled (gel characteristics)
        4. Stable (default liquid state)
        5. Unknown (insufficient data)
        """

        # Level 1: Check for Only Air
        only_air_check = self._check_only_air(det_summary, line_analysis)
        if only_air_check['is_only_air']:
            return {
                'vial_state': 'only_air',
                'confidence': only_air_check['confidence'],
                'reason': only_air_check['reason'],
                'metrics': only_air_check
            }

        # Level 2: Check if we have any liquid to analyze
        if not det_summary.has_liquid:
            # No liquid detections, check if analysis tools found anything
            if curve_analysis and curve_analysis.get('success') and \
                    curve_analysis.get('gelled_by_curve', False):
                return {
                    'vial_state': 'gelled',
                    'confidence': 'medium',
                    'reason': 'curve_analysis_without_detections',
                    'metrics': curve_analysis
                }

            return {
                'vial_state': 'unknown',
                'confidence': 'low',
                'reason': 'no_liquid_detections',
                'metrics': self._summary_to_dict(det_summary)
            }

        # Level 3: Check for Phase Separation
        phase_check = self._check_phase_separation(det_summary, line_analysis)
        if phase_check['is_phase_separated']:
            return {
                'vial_state': 'phase_separated',
                'confidence': phase_check['confidence'],
                'reason': phase_check['reason'],
                'metrics': phase_check
            }

        # Level 4: Check for Gel
        gel_check = self._check_gelled(det_summary, curve_analysis)
        if gel_check['is_gelled']:
            return {
                'vial_state': 'gelled',
                'confidence': gel_check['confidence'],
                'reason': gel_check['reason'],
                'metrics': gel_check
            }

        # Level 5: Default to Stable
        return {
            'vial_state': 'stable',
            'confidence': 'high',
            'reason': 'liquid_present_no_anomalies',
            'metrics': {
                'liquid_detections': len(det_summary.liquid_detections),
                'gel_count': det_summary.gel_count,
                'stable_count': det_summary.stable_count
            }
        }

    def _check_only_air(self,
                        det_summary: DetectionSummary,
                        line_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if vial contains only air (no liquid).

        Criteria:
        - No liquid detections OR
        - Large air region with horizontal line (meniscus) and minimal liquid
        """
        result = {
            'is_only_air': False,
            'confidence': 'low',
            'reason': '',
            'air_count': det_summary.air_count,
            'liquid_count': det_summary.gel_count + det_summary.stable_count,
            'max_air_height': det_summary.max_air_height_fraction
        }

        # Case 1: No liquid detections at all
        if not det_summary.has_liquid:
            # Check for meniscus line as supporting evidence
            if line_analysis and line_analysis.get('success'):
                h_positions = line_analysis.get('horizontal_positions', [])
                h_lengths = line_analysis.get('horizontal_lengths', [])

                # Look for long horizontal line in upper region
                min_line_len = ONLY_AIR_CLASSIFICATION['min_horizontal_line_len_frac']
                for pos, length in zip(h_positions, h_lengths):
                    if pos < 0.3 and length >= min_line_len:
                        result['is_only_air'] = True
                        result['confidence'] = 'high'
                        result['reason'] = 'no_liquid_with_meniscus_line'
                        result['meniscus_position'] = pos
                        return result

            # No meniscus line but still no liquid
            if det_summary.air_count > 0:
                result['is_only_air'] = True
                result['confidence'] = 'medium'
                result['reason'] = 'air_detected_no_liquid'
                return result

        # Case 2: Large air region with minimal liquid
        min_air_frac = ONLY_AIR_CLASSIFICATION['min_air_height_fraction']
        if det_summary.max_air_height_fraction >= min_air_frac:
            liquid_count = det_summary.gel_count + det_summary.stable_count

            # Air dominates and very little liquid
            if liquid_count <= 1:
                result['is_only_air'] = True
                result['confidence'] = 'medium'
                result['reason'] = 'air_dominant_minimal_liquid'
                return result

        result['reason'] = 'liquid_content_detected'
        return result

    def _check_phase_separation(self,
                                det_summary: DetectionSummary,
                                line_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for phase separation.

        Criteria (in priority order):
        1. Multiple horizontal lines (strong evidence)
        2. Large vertical gap between liquid detections
        3. Multiple distinct liquid regions
        """
        result = {
            'is_phase_separated': False,
            'confidence': 'low',
            'reason': ''
        }

        # Criterion 1: Multiple horizontal lines (strongest evidence)
        if line_analysis and line_analysis.get('success'):
            num_lines = line_analysis.get('num_horizontal_lines', 0)
            result['num_horizontal_lines'] = num_lines

            if num_lines >= 2:
                result['is_phase_separated'] = True
                result['confidence'] = 'high'
                result['reason'] = 'multiple_horizontal_phase_boundaries'
                result['line_positions'] = line_analysis.get('horizontal_positions', [])
                return result

        # Criterion 2: Large vertical gap
        gap_threshold = PHASE_SEPARATION_THRESHOLDS.get('gap_thr', 0.03)
        if det_summary.max_vertical_gap >= gap_threshold:
            result['is_phase_separated'] = True
            result['confidence'] = 'medium'
            result['reason'] = 'large_vertical_gap_between_liquid'
            result['max_gap'] = det_summary.max_vertical_gap
            return result

        # Criterion 3: Multiple distinct liquid regions
        if len(det_summary.liquid_detections) >= 2:
            # Check if detections are truly separated (not just multiple boxes)
            sorted_dets = sorted(det_summary.liquid_detections,
                                 key=lambda d: d['center_y'])

            # If we have detections with some gap, consider it
            if det_summary.max_vertical_gap > gap_threshold * 0.5:
                result['is_phase_separated'] = True
                result['confidence'] = 'low'
                result['reason'] = 'multiple_liquid_regions_with_gap'
                result['num_regions'] = len(det_summary.liquid_detections)
                return result

        result['reason'] = 'no_separation_detected'
        return result

    def _check_gelled(self,
                      det_summary: DetectionSummary,
                      curve_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for gelled state.

        Criteria (in priority order):
        1. Curve analysis indicates gel (strongest evidence)
        2. Gel detections dominate by area
        3. Gel detections dominate by count
        """
        result = {
            'is_gelled': False,
            'confidence': 'low',
            'reason': '',
            'gel_count': det_summary.gel_count,
            'stable_count': det_summary.stable_count,
            'gel_area_fraction': det_summary.gel_area_fraction
        }

        # Criterion 1: Curve analysis (strongest evidence)
        if curve_analysis and curve_analysis.get('success'):
            if curve_analysis.get('gelled_by_curve', False):
                result['is_gelled'] = True
                result['confidence'] = 'high'
                result['reason'] = 'curve_variance_indicates_gel'
                result['curve_variance'] = curve_analysis.get('variance_from_baseline', 0.0)
                return result

        # Criterion 2: Gel dominance by area
        gel_area_threshold = DETECTION_THRESHOLDS.get('gel_area_frac', 0.35)
        if det_summary.gel_area_fraction >= gel_area_threshold:
            result['is_gelled'] = True
            result['confidence'] = 'high'
            result['reason'] = 'gel_dominant_by_area'
            return result

        # Criterion 3: Gel dominance by count
        min_gel_count = DETECTION_THRESHOLDS.get('gel_dominance_count', 1)
        if det_summary.gel_count >= min_gel_count and \
                det_summary.gel_count > det_summary.stable_count:
            result['is_gelled'] = True
            result['confidence'] = 'medium'
            result['reason'] = 'gel_dominant_by_count'
            return result

        result['reason'] = 'insufficient_gel_evidence'
        return result

    def _summary_to_dict(self, summary: DetectionSummary) -> Dict[str, Any]:
        """Convert DetectionSummary to dictionary for serialization."""
        return {
            'total_detections': len(summary.all_detections),
            'liquid_detections': len(summary.liquid_detections),
            'gel_count': summary.gel_count,
            'stable_count': summary.stable_count,
            'air_count': summary.air_count,
            'cap_count': summary.cap_count,
            'gel_area_fraction': summary.gel_area_fraction,
            'max_air_height_fraction': summary.max_air_height_fraction,
            'max_vertical_gap': summary.max_vertical_gap
        }

    def _error_result(self, reason: str, path: Path) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'vial_state': 'unknown',
            'confidence': 'none',
            'reason': reason,
            'error': True,
            'path': str(path)
        }


# Convenience function for backward compatibility
def classify_vial_state(crop_path: Path,
                        label_path: Path,
                        use_line_detection: bool = True,
                        use_curved_line_detection: bool = True,
                        region_exclusion: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Classify vial state (backward compatible interface)."""
    classifier = VialStateClassifier(
        use_line_detection=use_line_detection,
        use_curved_line_detection=use_curved_line_detection,
        region_exclusion=region_exclusion
    )
    return classifier.classify(crop_path, label_path)