"""
Vial state classification module.
Determines vial state: stable, gelled, phase_separated, or only_air.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from config import CLASS_IDS, LIQUID_CLASSES, DETECTION_THRESHOLDS, PHASE_SEPARATION_THRESHOLDS
from robotlab_utils.bbox_utils import (
    yolo_line_to_xyxy, box_area, merge_detections_by_iou,
    filter_detections_by_region, merge_detections_by_iou_with_priority
)
from analysis.turbidity_analysis import compute_turbidity_profile, detect_turbidity_peaks


class VialStateClassifier:
    """Classifier for determining vial state from detections."""
    
    def __init__(self, use_turbidity: bool = False, merge_boxes: bool = True):
        """
        Initialize classifier.
        
        Args:
            use_turbidity: Whether to use turbidity analysis for phase separation
            merge_boxes: Whether to merge overlapping detection boxes
        """
        self.use_turbidity = use_turbidity
        self.merge_boxes = merge_boxes
    
    def classify(self, crop_path: Path, label_path: Path) -> Dict[str, Any]:
        """
        Classify vial state from crop image and detections.
        
        Args:
            crop_path: Path to crop image
            label_path: Path to YOLO label file
            
        Returns:
            Dictionary with classification results
        """
        # Load image
        img = cv2.imread(str(crop_path))
        if img is None:
            return {
                "vial_state": "unknown",
                "reason": "crop_not_found"
            }
        
        H, W = img.shape[:2]
        
        # Handle no detection file
        if not label_path.exists():
            return self._classify_no_detections(img)
        
        # Parse detections
        detections = self._parse_detections(label_path, W, H)
        
        if not detections:
            return self._classify_no_detections(img)
        
        # Merge overlapping boxes if enabled
        if self.merge_boxes:
            detections = merge_detections_by_iou(
                detections,
                iou_thr=DETECTION_THRESHOLDS['iou_thr']
            )
        
        # Separate liquid and non-liquid detections
        liquid_dets = [d for d in detections if self._is_liquid(d['class_id'])]
        
        if not liquid_dets:
            return {
                "vial_state": "only_air",
                "reason": "no_liquid_detected",
                "total_detections": len(detections)
            }
        
        # Check for phase separation
        is_separated, phase_metrics = self._detect_phase_separation(
            liquid_dets, H, W, img
        )
        
        if is_separated:
            return {
                "vial_state": "phase_separated",
                "phase_metrics": phase_metrics,
                "detections": self._summarize_detections(liquid_dets)
            }
        
        # Classify as gelled or stable
        state = self._classify_liquid_state(liquid_dets, H, W)
        
        return {
            "vial_state": state,
            "detections": self._summarize_detections(liquid_dets),
            "coverage": self._calculate_coverage(liquid_dets, H, W)
        }
    
    def _parse_detections(self, label_path: Path, W: int, H: int) -> List[Dict[str, Any]]:
        """Parse YOLO label file into detection dictionaries."""
        detections = []
        
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parsed = yolo_line_to_xyxy(line, W, H)
                if not parsed:
                    continue
                
                cls_id, box, conf = parsed
                
                # Filter by minimum confidence
                if conf < DETECTION_THRESHOLDS['conf_min']:
                    continue
                
                det = {
                    'class_id': cls_id,
                    'box': box,
                    'confidence': conf,
                    'center_y': (box[1] + box[3]) / 2.0,
                    'area': box_area(box)
                }
                detections.append(det)
        
        return detections
    
    def _is_liquid(self, class_id: int) -> bool:
        """Check if class ID corresponds to liquid."""
        return class_id in LIQUID_CLASSES
    
    def _classify_no_detections(self, img: np.ndarray) -> Dict[str, Any]:
        """Classify when no detections are present."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # High brightness with low variance suggests empty vial
        if mean_brightness > 150 and brightness_std < 30:
            state = "only_air"
            reason = "no_detections_appears_empty"
        else:
            state = "unknown"
            reason = "detection_failed"
        
        return {
            "vial_state": state,
            "reason": reason,
            "brightness_metrics": {
                "mean": float(mean_brightness),
                "std": float(brightness_std)
            }
        }
    
    def _detect_phase_separation(self, liquid_dets: List[Dict[str, Any]], 
                                 H: int, W: int, 
                                 img: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if vial has phase separation.
        
        Returns:
            Tuple of (is_separated, metrics)
        """
        # Method 1: Multiple liquid detection regions
        if len(liquid_dets) >= 2:
            return True, {'method': 'multiple_regions', 'count': len(liquid_dets)}
        
        # Method 2: Check vertical gaps between detections
        if len(liquid_dets) > 1:
            sorted_dets = sorted(liquid_dets, key=lambda d: d['center_y'])
            
            for i in range(len(sorted_dets) - 1):
                current_bottom = sorted_dets[i]['box'][3]
                next_top = sorted_dets[i + 1]['box'][1]
                gap = (next_top - current_bottom) / H
                
                if gap >= PHASE_SEPARATION_THRESHOLDS['gap_thr']:
                    return True, {'method': 'vertical_gap', 'gap': gap}
        
        # Method 3: Check vertical span
        if len(liquid_dets) > 1:
            y_positions = [d['center_y'] for d in liquid_dets]
            span = (max(y_positions) - min(y_positions)) / H
            
            if span >= PHASE_SEPARATION_THRESHOLDS['span_thr']:
                return True, {'method': 'vertical_span', 'span': span}
        
        # Method 4: Turbidity analysis (if enabled and image available)
        if self.use_turbidity and img is not None:
            profile = compute_turbidity_profile(img)
            is_separated, turbidity_metrics = detect_turbidity_peaks(profile)
            
            if is_separated:
                return True, {'method': 'turbidity', **turbidity_metrics}
        
        return False, {'method': 'none'}
    
    def _classify_liquid_state(self, liquid_dets: List[Dict[str, Any]], 
                               H: int, W: int) -> str:
        """
        Classify liquid as gelled or stable.
        
        Returns:
            'gelled' or 'stable'
        """
        total_area = sum(d['area'] for d in liquid_dets)
        gel_area = sum(d['area'] for d in liquid_dets 
                      if d['class_id'] == CLASS_IDS['GEL'])
        
        n_gel = sum(1 for d in liquid_dets if d['class_id'] == CLASS_IDS['GEL'])
        n_stable = sum(1 for d in liquid_dets if d['class_id'] == CLASS_IDS['STABLE'])
        
        # Check gel area fraction
        if total_area > 0:
            gel_fraction = gel_area / total_area
            if gel_fraction >= DETECTION_THRESHOLDS['gel_area_frac']:
                return 'gelled'
        
        # Check gel dominance by count
        if (n_gel - n_stable) >= DETECTION_THRESHOLDS['gel_dominance_count']:
            return 'gelled'
        
        return 'stable'
    
    def _summarize_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics of detections."""
        n_gel = sum(1 for d in detections if d['class_id'] == CLASS_IDS['GEL'])
        n_stable = sum(1 for d in detections if d['class_id'] == CLASS_IDS['STABLE'])
        
        total_area = sum(d['area'] for d in detections)
        gel_area = sum(d['area'] for d in detections 
                      if d['class_id'] == CLASS_IDS['GEL'])
        
        return {
            'n_gel': n_gel,
            'n_stable': n_stable,
            'total_count': len(detections),
            'gel_area_fraction': gel_area / total_area if total_area > 0 else 0
        }
    
    def _calculate_coverage(self, detections: List[Dict[str, Any]], 
                           H: int, W: int) -> float:
        """Calculate liquid coverage of vial."""
        total_liquid_area = sum(d['area'] for d in detections)
        vial_area = H * W
        return total_liquid_area / vial_area if vial_area > 0 else 0