"""
Enhanced state classifier with integrated image analysis tools.
Combines detection-based classification with line detection and curve analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config import (
    CLASS_IDS, LIQUID_CLASSES, DETECTION_THRESHOLDS,
    ONLY_AIR_CLASSIFICATION, CURVE_PARAMS, LINE_PARAMS, REGION_EXCLUSION, PHASE_SEPARATION_THRESHOLDS
)
from robotlab_utils.io_utils import read_yolo_labels
from analysis.phase_separation import PhaseSeparationDetector
from analysis.gelled_analysis import run_curve_metrics
from image_analysis.line_hv_detection import LineDetector


class VialStateClassifier:
    """
    Classifier that integrates multiple analysis methods:
    1. YOLO detection-based classification
    2. Horizontal line detection for phase separation
    3. Curved line analysis for gel detection
    4. Only-air classification for empty vials
    """

    def __init__(self,
                 use_line_detection: bool = True,
                 use_curved_line_detection: bool = True,
                 use_turbidity: bool = False,
                 merge_boxes: bool = True,
                 region_exclusion: Optional[Dict[str, float]] = None):
        """
        Initialize classifier.

        Args:
            use_line_detection: Enable horizontal line detection
            use_curved_line_detection: Enable curved line analysis
            use_turbidity: Enable turbidity analysis
            merge_boxes: Merge overlapping detection boxes
            region_exclusion: Region exclusion config for phase separation
        """
        self.use_line_detection = use_line_detection
        self.use_curved_line_detection = use_curved_line_detection
        self.use_turbidity = use_turbidity
        self.merge_boxes = merge_boxes
        self.region_exclusion = region_exclusion or REGION_EXCLUSION

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

        Args:
            crop_path: Path to crop image
            label_path: Path to YOLO label file
            save_analysis_viz: Save visualization from analysis tools
            viz_output_dir: Directory to save visualizations

        Returns:
            Dictionary with classification results and metrics
        """
        # Load image
        img = cv2.imread(str(crop_path))
        if img is None:
            return {"vial_state": "unknown", "reason": "image_load_failed"}

        H, W = img.shape[:2]

        # Parse YOLO detections
        detections = read_yolo_labels(label_path)
        detection_dict = self._prepare_detections(detections, H, W)

        # Initialize analysis results
        analysis_metrics = {
            "num_detections": len(detections),
            "detection_classes": self._count_detection_classes(detections),
        }

        # Line Detection Analysis
        line_analysis = None
        if self.use_line_detection and self.line_detector:
            line_analysis = self._analyze_lines(
                crop_path, H, W,
                save_analysis_viz, viz_output_dir
            )
            analysis_metrics["line_detection"] = line_analysis

        # Curve Analysis
        curve_analysis = None
        if self.use_curved_line_detection:
            curve_analysis = self._analyze_curve(
                crop_path,
                save_analysis_viz, viz_output_dir
            )
            analysis_metrics["curve_analysis"] = curve_analysis

        # Integrated Classification
        classification = self._integrated_classification(
            detection_dict,
            line_analysis,
            curve_analysis,
            H, W
        )

        # Combine results
        result = {
            **classification,
            "analysis_metrics": analysis_metrics
        }

        return result

    def _prepare_detections(self,
                           detections: List[Dict[str, Any]],
                           height: int,
                           width: int) -> Dict[str, Any]:
        """Convert YOLO detections to pixel coordinates and compute properties."""
        processed_dets = []

        for det in detections:
            # Convert from normalized to pixel coordinates
            cx = det['cx'] * width
            cy = det['cy'] * height
            w = det['w'] * width
            h = det['h'] * height

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

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
            processed_dets.append(processed)

        return {
            'detections': processed_dets,
            'height': height,
            'width': width
        }

    def _count_detection_classes(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count detections by class."""
        counts = {
            'gel': 0,
            'stable': 0,
            'air': 0,
            'cap': 0
        }

        for det in detections:
            class_id = det['class_id']
            if class_id == CLASS_IDS['GEL']:
                counts['gel'] += 1
            elif class_id == CLASS_IDS['STABLE']:
                counts['stable'] += 1
            elif class_id == CLASS_IDS['AIR']:
                counts['air'] += 1
            elif class_id == CLASS_IDS['CAP']:
                counts['cap'] += 1

        return counts

    def _analyze_lines(self,
                      crop_path: Path,
                      height: int,
                      width: int,
                      save_viz: bool = False,
                      viz_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze horizontal and vertical lines."""
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

        # Extract key metrics
        h_lines = result['horizontal_lines']['lines']
        v_lines = result['vertical_lines']['lines']

        return {
            'num_horizontal_lines': len(h_lines),
            'num_vertical_lines': len(v_lines),
            'horizontal_positions': [l['y_position'] for l in h_lines],
            'vertical_boundaries': {
                'left': result['vertical_lines'].get('left_boundary', 0.0),
                'right': result['vertical_lines'].get('right_boundary', 1.0)
            },
            'has_phase_boundary': len(h_lines) >= 2
        }

    def _analyze_curve(self,
                      crop_path: Path,
                      save_viz: bool = False,
                      viz_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze curve characteristics for gel detection."""
        curve_result = run_curve_metrics(crop_path)

        # Save visualization if requested
        if save_viz and viz_dir and curve_result.get('stats'):
            from image_analysis.guided_curve_tracer import GuidedCurveTracer

            viz_dir.mkdir(parents=True, exist_ok=True)

            # Load image
            img = cv2.imread(str(crop_path))
            if img is not None:
                # Initialize tracer with same params as curve analysis
                tracer = GuidedCurveTracer(
                    vertical_bounds=CURVE_PARAMS.get("vertical_bounds", (0.30, 0.80)),
                    horizontal_bounds=CURVE_PARAMS.get("horizontal_bounds", (0.05, 0.95)),
                    search_offset_frac=CURVE_PARAMS.get("search_offset_frac", 0.10),
                    median_kernel=CURVE_PARAMS.get("median_kernel", 9),
                    max_step_px=CURVE_PARAMS.get("max_step_px", 4)
                )

                # Trace curve
                xs, ys, metadata = tracer.trace_curve(img, crop_path, guide_y=None)

                # Save visualization
                if len(xs) > 0:
                    viz_path = viz_dir / f"{crop_path.stem}_curve_analysis.png"
                    tracer.visualize(img, xs, ys, metadata, viz_path)

        # Extract key metrics
        stats = curve_result.get('stats', {})
        return {
            'gelled_by_curve': curve_result.get('gelled_by_curve', False),
            'variance_from_baseline': stats.get('variance_from_baseline', 0.0),
            'std_dev_from_baseline': stats.get('std_dev_from_baseline', 0.0),
            'roughness': stats.get('roughness', 0.0),
            'num_points': stats.get('num_points', 0),
            'reason': curve_result.get('reason', 'no_analysis')
        }

    def _integrated_classification(self,
                                  detection_dict: Dict[str, Any],
                                  line_analysis: Optional[Dict[str, Any]],
                                  curve_analysis: Optional[Dict[str, Any]],
                                  height: int,
                                  width: int) -> Dict[str, Any]:
        """
        Integrated classification using all available information.

        Classification hierarchy:
        1. Only air (special case - no liquid detected)
        2. Gelled (curve analysis + detection patterns)
        3. Phase separated (horizontal lines + detection patterns)
        4. Stable (default liquid state)
        """
        detections = detection_dict['detections']

        # Count detections by class
        class_counts = self._count_detection_classes(
            [{'class_id': d['class_id']} for d in detections]
        )

        # === CASE 1: Only Air Classification ===
        only_air_result = self._classify_only_air(
            detections, class_counts, line_analysis, height, width
        )
        if only_air_result['is_only_air']:
            return {
                'vial_state': 'only_air',
                **only_air_result
            }

        # Filter liquid detections (exclude air and cap)
        liquid_dets = [d for d in detections if d['class_id'] in LIQUID_CLASSES]

        # === CASE 2: No Liquid Detections ===
        if len(liquid_dets) == 0:
            # Check if we have strong curve or line evidence
            if curve_analysis and curve_analysis.get('gelled_by_curve', False):
                return {
                    'vial_state': 'gelled',
                    'reason': 'curve_analysis_without_detections',
                    'confidence': 'medium',
                    **curve_analysis
                }

            return {
                'vial_state': 'unknown',
                'reason': 'no_liquid_detections',
                'class_counts': class_counts
            }

        # === CASE 3: Gelled Classification ===
        gel_result = self._classify_gelled(
            liquid_dets, class_counts, curve_analysis
        )
        if gel_result['is_gelled']:
            return {
                'vial_state': 'gelled',
                **gel_result
            }

        # === CASE 4: Phase Separated Classification ===
        phase_result = self._classify_phase_separated(
            detections, line_analysis, height, width
        )
        if phase_result['is_phase_separated']:
            return {
                'vial_state': 'phase_separated',
                **phase_result
            }

        # === CASE 5: Stable (Default) ===
        return {
            'vial_state': 'stable',
            'reason': 'default_stable_state',
            'liquid_detections': len(liquid_dets),
            'class_counts': class_counts
        }

    def _classify_only_air(self,
                          detections: List[Dict[str, Any]],
                          class_counts: Dict[str, int],
                          line_analysis: Optional[Dict[str, Any]],
                          height: int,
                          width: int) -> Dict[str, Any]:
        """
        Classify as only air based on:
        1. Predominant air detections
        2. Large air region height
        3. Presence of long horizontal lines (meniscus)
        """
        air_count = class_counts['air']
        liquid_count = class_counts['gel'] + class_counts['stable']

        # Find air detections
        air_dets = [d for d in detections if d['class_id'] == CLASS_IDS['AIR']]

        metrics = {
            'is_only_air': False,
            'reason': '',
            'air_count': air_count,
            'liquid_count': liquid_count
        }

        # No air detections
        if air_count == 0:
            # Check if we have a very long horizontal line at the top
            # This could indicate a meniscus with no liquid visible
            if line_analysis:
                h_lines = line_analysis.get('horizontal_positions', [])
                if len(h_lines) >= 1:
                    # Check if line is in upper region
                    top_line = min(h_lines)
                    if top_line < 0.3:  # Line in top 30% of image
                        metrics['is_only_air'] = True
                        metrics['reason'] = 'horizontal_line_meniscus_no_liquid'
                        metrics['meniscus_position'] = top_line
                        return metrics

            metrics['reason'] = 'no_air_detections'
            return metrics

        # Calculate air region height
        max_air_height = 0.0
        for air_det in air_dets:
            h_frac = air_det['height'] / height
            max_air_height = max(max_air_height, h_frac)

        metrics['max_air_height_fraction'] = max_air_height

        # Check if air dominates
        min_air_frac = ONLY_AIR_CLASSIFICATION['min_air_height_fraction']
        air_dominant = max_air_height >= min_air_frac

        # Check for horizontal line (meniscus indicator)
        has_meniscus_line = False
        meniscus_length = 0.0

        if line_analysis:
            h_lines = line_analysis.get('horizontal_positions', [])
            min_line_len = ONLY_AIR_CLASSIFICATION['min_horizontal_line_len_frac']

            # Check for long horizontal lines in analysis
            if len(h_lines) >= 1:
                # Get the topmost line (likely meniscus)
                top_line_y = min(h_lines)

                # Check line length from line_analysis details
                # Assuming we have access to full line data
                has_meniscus_line = True
                meniscus_length = min_line_len  # Placeholder

        metrics['has_meniscus_line'] = has_meniscus_line
        metrics['meniscus_length_fraction'] = meniscus_length

        # Decision logic
        if air_dominant and liquid_count == 0:
            metrics['is_only_air'] = True
            metrics['reason'] = 'air_dominant_no_liquid'
        elif air_dominant and has_meniscus_line:
            metrics['is_only_air'] = True
            metrics['reason'] = 'air_dominant_with_meniscus'
        elif air_count > 0 and liquid_count == 0 and has_meniscus_line:
            metrics['is_only_air'] = True
            metrics['reason'] = 'meniscus_only_no_liquid'
        else:
            metrics['reason'] = 'contains_liquid_or_insufficient_evidence'

        return metrics

    def _classify_gelled(self,
                        liquid_dets: List[Dict[str, Any]],
                        class_counts: Dict[str, int],
                        curve_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify as gelled based on:
        1. Curve variance analysis
        2. Gel detection dominance
        3. Combined heuristics
        """
        gel_count = class_counts['gel']
        stable_count = class_counts['stable']

        metrics = {
            'is_gelled': False,
            'reason': '',
            'gel_count': gel_count,
            'stable_count': stable_count
        }

        # Curve analysis check
        curve_indicates_gel = False
        if curve_analysis:
            curve_indicates_gel = curve_analysis.get('gelled_by_curve', False)
            metrics['curve_variance'] = curve_analysis.get('variance_from_baseline', 0.0)
            metrics['curve_reason'] = curve_analysis.get('reason', '')

        # Detection-based check
        total_liquid = gel_count + stable_count
        if total_liquid > 0:
            gel_fraction = gel_count / total_liquid
            metrics['gel_fraction'] = float(gel_fraction)

            # Calculate gel area fraction
            gel_dets = [d for d in liquid_dets if d['class_id'] == CLASS_IDS['GEL']]
            total_area = sum(d['area'] for d in liquid_dets)
            gel_area = sum(d['area'] for d in gel_dets)
            gel_area_frac = gel_area / total_area if total_area > 0 else 0.0
            metrics['gel_area_fraction'] = float(gel_area_frac)

        # Decision logic
        gel_area_thr = DETECTION_THRESHOLDS.get('gel_area_frac', 0.35)

        if curve_indicates_gel:
            metrics['is_gelled'] = True
            metrics['reason'] = 'curve_analysis_indicates_gel'
            metrics['confidence'] = 'high'
        elif gel_count > 0 and metrics.get('gel_area_fraction', 0) >= gel_area_thr:
            metrics['is_gelled'] = True
            metrics['reason'] = 'gel_detection_dominant_by_area'
            metrics['confidence'] = 'high'
        elif gel_count > stable_count and gel_count >= DETECTION_THRESHOLDS.get('gel_dominance_count', 1):
            metrics['is_gelled'] = True
            metrics['reason'] = 'gel_detection_dominant_by_count'
            metrics['confidence'] = 'medium'
        else:
            metrics['reason'] = 'insufficient_gel_evidence'

        return metrics

    def _classify_phase_separated(self,
                                  detections: List[Dict[str, Any]],
                                  line_analysis: Optional[Dict[str, Any]],
                                  height: int,
                                  width: int) -> Dict[str, Any]:
        """
        Classify as phase separated based on:
        1. Multiple horizontal lines
        2. Multiple liquid regions
        3. Vertical gaps in detections
        """
        metrics = {
            'is_phase_separated': False,
            'reason': ''
        }

        # Line analysis check
        if line_analysis:
            num_h_lines = line_analysis.get('num_horizontal_lines', 0)
            metrics['num_horizontal_lines'] = num_h_lines

            if num_h_lines >= 2:
                metrics['is_phase_separated'] = True
                metrics['reason'] = 'multiple_horizontal_lines_detected'
                metrics['confidence'] = 'high'
                metrics['line_positions'] = line_analysis.get('horizontal_positions', [])
                return metrics

        # Detection-based check using phase separation detector
        liquid_dets = [d for d in detections if d['class_id'] in LIQUID_CLASSES]

        if len(liquid_dets) >= 2:
            # Check for multiple separated regions
            sorted_dets = sorted(liquid_dets, key=lambda d: d['center_y'])

            # Calculate gaps
            gaps = []
            for i in range(len(sorted_dets) - 1):
                bottom = sorted_dets[i]['box'][3]
                top = sorted_dets[i + 1]['box'][1]
                gap = (top - bottom) / height
                gaps.append(gap)

            max_gap = max(gaps) if gaps else 0.0
            metrics['max_vertical_gap'] = float(max_gap)

            gap_threshold = PHASE_SEPARATION_THRESHOLDS.get('gap_thr', 0.03)
            if max_gap >= gap_threshold:
                metrics['is_phase_separated'] = True
                metrics['reason'] = 'vertical_gap_between_liquid_regions'
                metrics['confidence'] = 'medium'
                return metrics

        metrics['reason'] = 'no_separation_evidence'
        return metrics


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