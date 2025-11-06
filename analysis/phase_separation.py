"""
Phase separation detection module.
Analyzes liquid detections to determine if vial exhibits phase separation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from config import PHASE_SEPARATION_THRESHOLDS, CLASS_IDS, LIQUID_CLASSES, LINE_PARAMS, REGION_EXCLUSION
from image_analysis.line_hv_detection import LineDetector
from pathlib import Path


class PhaseSeparationDetector:
    """Detector for phase separation in vials."""
    
    def __init__(self, 
                 gap_threshold: float = PHASE_SEPARATION_THRESHOLDS['gap_thr'],
                 span_threshold: float = PHASE_SEPARATION_THRESHOLDS['span_thr'],
                 min_area_fraction: float = PHASE_SEPARATION_THRESHOLDS['min_area_frac']):
        """
        Initialize detector with thresholds.
        
        Args:
            gap_threshold: Minimum vertical gap between layers (normalized)
            span_threshold: Minimum vertical span for separation (normalized)
            min_area_fraction: Minimum area fraction for valid detection
        """
        self.gap_threshold = gap_threshold
        self.span_threshold = span_threshold
        self.min_area_fraction = min_area_fraction
    
    def detect(self, 
               detections: List[Dict[str, Any]], 
               image_height: int,
               image_width: int,
               cap_bottom_y: Optional[float] = None,
               image_path: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect phase separation from liquid detections.
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            image_height: Height of image in pixels
            image_width: Width of image in pixels
            cap_bottom_y: Optional cap bottom position
            
        Returns:
            Tuple of (is_separated, analysis_metrics)
        """
        # Keep only liquid detections
        liquid_dets = self._filter_liquid_detections(
            detections, image_height, image_width, cap_bottom_y
        )
        if len(liquid_dets) < 1:
            return False, {"reason": "no_liquid_detections"}

        # Sort liquid detections by vertical center
        liquid_dets = sorted(liquid_dets, key=lambda d: d["center_y"])

        # Count-based heuristic (multiple liquid regions from detector)
        if len(liquid_dets) >= 2:
            metrics = {
                "method": "multiple_liquid_regions",
                "num_regions": len(liquid_dets),
            }
            return True, metrics

        # Horizontal-line detection
        horiz_metrics: Dict[str, Any] = {}

        if image_path is not None:
            detector = LineDetector(
                min_line_length=LINE_PARAMS["min_line_length"],
                merge_threshold_horizontal=LINE_PARAMS["merge_threshold_horizontal"],
                merge_threshold_vertical=LINE_PARAMS["merge_threshold_vertical"]
            )
            line_result = detector.detect(
                image_path=Path(image_path),
                top_exclusion=REGION_EXCLUSION["top_fraction"],
                bottom_exclusion=REGION_EXCLUSION["bottom_fraction"],
            )

            h_lines = line_result["horizontal_lines"]["lines"]
            num_hlines = int(line_result["horizontal_lines"]["num_lines"])
            y_norms = [float(l["y_position"]) for l in h_lines]
            y_pixels = [int(round(y * image_height)) for y in y_norms]

            horiz_metrics = {
                "method": "horizontal_lines",
                "num_horizontal_lines": num_hlines,
                "y_norm": y_norms,
                "y_pixels": y_pixels,
                "left_boundary_norm": float(line_result["vertical_lines"]["left_boundary"]),
                "right_boundary_norm": float(line_result["vertical_lines"]["right_boundary"]),
            }

            # phase separation
            if num_hlines >= 2:
                return True, horiz_metrics

        # Gap-based check between top/bottom of the liquid region
        is_sep, gap_metrics = self._check_vertical_gaps(liquid_dets, image_height)
        if is_sep:
            gap_metrics.update(horiz_metrics)
            return True, gap_metrics

        # Span-based check (stratification)
        is_sep, span_metrics = self._check_vertical_span(liquid_dets, image_height)
        if is_sep:
            span_metrics.update(horiz_metrics)
            return True, span_metrics

        # Layer analysis
        is_sep, layer_metrics = self._analyze_layers(liquid_dets, image_height)
        if is_sep:
            layer_metrics.update(horiz_metrics)
            return True, layer_metrics

        # Nothing triggered
        return False, {"reason": "no_separation_detected", **horiz_metrics}

    
    def _filter_liquid_detections(self,
                                  detections: List[Dict[str, Any]],
                                  height: int,
                                  width: int,
                                  cap_bottom_y: Optional[float]) -> List[Dict[str, Any]]:
        """Filter detections to valid liquid regions."""
        total_area = float(height * width)
        filtered = []
        
        for det in detections:
            # Check if liquid class
            if det['class_id'] not in LIQUID_CLASSES:
                continue
            
            # Check minimum confidence (assumed to be pre-filtered)
            # if det['confidence'] < 0.5:
            #     continue
            
            # Check if below cap
            # if cap_bottom_y is not None and det['box'][1] <= cap_bottom_y:
            #     continue
            
            # Check minimum area
            if (det['area'] / total_area) < self.min_area_fraction:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _check_vertical_gaps(self, 
                            detections: List[Dict[str, Any]], 
                            height: int) -> Tuple[bool, Dict[str, Any]]:
        """Check for significant vertical gaps between detections."""
        if len(detections) < 2:
            return False, {}
        
        gaps = []
        for i in range(len(detections) - 1):
            current_bottom = detections[i]['box'][3]
            next_top = detections[i + 1]['box'][1]
            gap = (next_top - current_bottom) / height
            gaps.append(gap)
            
            if gap >= self.gap_threshold:
                return True, {
                    'method': 'vertical_gap',
                    'gap_value': float(gap),
                    'gap_index': i,
                    'all_gaps': gaps
                }
        
        return False, {'gaps': gaps}
    
    def _check_vertical_span(self, 
                           detections: List[Dict[str, Any]], 
                           height: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if detections span significant vertical distance."""
        if len(detections) < 2:
            return False, {}
        
        y_positions = [d['center_y'] for d in detections]
        span = (max(y_positions) - min(y_positions)) / height
        
        if span >= self.span_threshold:
            return True, {
                'method': 'vertical_span',
                'span_value': float(span),
                'min_y': float(min(y_positions) / height),
                'max_y': float(max(y_positions) / height)
            }
        
        return False, {'span': float(span)}
    
    def _analyze_layers(self,
                       detections: List[Dict[str, Any]],
                       height: int) -> Tuple[bool, Dict[str, Any]]:
        """Analyze detections for distinct layer formation."""
        if len(detections) < 2:
            return False, {}
        
        # Group detections into potential layers
        layer_tolerance = height * 0.05  # 5% of height
        layers = []
        current_layer = [detections[0]]
        
        for i in range(1, len(detections)):
            det = detections[i]
            prev_det = current_layer[-1]
            
            # Check if part of same layer
            if abs(det['center_y'] - prev_det['center_y']) < layer_tolerance:
                current_layer.append(det)
            else:
                layers.append(current_layer)
                current_layer = [det]
        
        # Add last layer
        if current_layer:
            layers.append(current_layer)
        
        # Check if we have multiple distinct layers
        if len(layers) >= 2:
            # Analyze each layer
            layer_info = []
            for idx, layer in enumerate(layers):
                # Determine dominant class in layer
                gel_count = sum(1 for d in layer if d['class_id'] == CLASS_IDS['GEL'])
                stable_count = sum(1 for d in layer if d['class_id'] == CLASS_IDS['STABLE'])
                
                dominant = 'gel' if gel_count > stable_count else 'stable'
                
                layer_info.append({
                    'layer_index': idx,
                    'num_detections': len(layer),
                    'dominant_class': dominant,
                    'center_y_normalized': np.mean([d['center_y'] for d in layer]) / height,
                    'total_area': sum(d['area'] for d in layer)
                })
            
            return True, {
                'method': 'layer_analysis',
                'num_layers': len(layers),
                'layers': layer_info
            }
        
        return False, {'num_layers': len(layers)}
    
    def analyze_separation_characteristics(self,
                                          detections: List[Dict[str, Any]],
                                          height: int,
                                          width: int) -> Dict[str, Any]:
        """
        Provide detailed analysis of phase separation characteristics.
        
        Args:
            detections: Liquid detection list
            height: Image height
            width: Image width
            
        Returns:
            Dictionary with detailed analysis
        """
        if not detections:
            return {'error': 'no_detections'}
        
        # Calculate various metrics
        total_area = sum(d['area'] for d in detections)
        coverage = total_area / (height * width)
        
        # Vertical distribution
        y_positions = [d['center_y'] for d in detections]
        vertical_distribution = {
            'mean_y': np.mean(y_positions) / height,
            'std_y': np.std(y_positions) / height,
            'min_y': min(y_positions) / height,
            'max_y': max(y_positions) / height,
            'span': (max(y_positions) - min(y_positions)) / height
        }
        
        # Class distribution
        gel_dets = [d for d in detections if d['class_id'] == CLASS_IDS['GEL']]
        stable_dets = [d for d in detections if d['class_id'] == CLASS_IDS['STABLE']]
        
        class_distribution = {
            'gel_count': len(gel_dets),
            'stable_count': len(stable_dets),
            'gel_area': sum(d['area'] for d in gel_dets),
            'stable_area': sum(d['area'] for d in stable_dets),
            'gel_fraction': sum(d['area'] for d in gel_dets) / total_area if total_area > 0 else 0
        }
        
        return {
            'total_detections': len(detections),
            'coverage': float(coverage),
            'vertical_distribution': vertical_distribution,
            'class_distribution': class_distribution
        }
