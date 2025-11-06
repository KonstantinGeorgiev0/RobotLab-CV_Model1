"""
Vial state classification module.
Determines vial state: stable, gelled, phase_separated, or only_air.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from config import *
from robotlab_utils.bbox_utils import (
    yolo_line_to_xyxy, box_area, filter_detections_by_exclusion_region,
)
from analysis.phase_separation import PhaseSeparationDetector
from analysis.gelled_analysis import run_curve_metrics
from analysis.turbidity_analysis import compute_turbidity_profile, detect_turbidity_peaks
from image_analysis.line_hv_detection import LineDetector
from yolov5.utils.general import LOGGER


class VialStateClassifier:
    """Classifier for determining vial state from detections."""
    
    def __init__(
            self, use_line_detection: bool = False, use_curved_line_detection: bool = False,
            use_turbidity: bool = False, merge_boxes: bool = True,
            region_exclusion: Optional[Dict[str, float]] = None
    ):
        """
        Initialize classifier.
        
        Args:
            use_turbidity: Whether to use turbidity analysis for phase separation
            merge_boxes: Whether to merge overlapping detection boxes
        """
        self.use_line_detection = use_line_detection
        self.use_curved_line_detection = use_curved_line_detection
        self.use_turbidity = use_turbidity
        self.merge_boxes = merge_boxes
        self.region_exclusion = region_exclusion or {}
    
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

        # Count detections in original label file
        original_detection_count = 0
        with open(label_path) as f:
            for line in f:
                if line.strip():
                    original_detection_count += 1

        # Parse detections
        detections = self._parse_detections(label_path, W, H)

        # Create region info for output
        region_info = {
            "enabled": bool(self.region_exclusion),
            "original_detections": original_detection_count,
            "after_exclusion": len(detections),
            "excluded_count": original_detection_count - len(detections)
        }

        # Filter detections by exclusion region
        if self.region_exclusion:
            top_frac = self.region_exclusion.get('top_fraction', 0.0)
            bottom_frac = self.region_exclusion.get('bottom_fraction', 0.0)
            region_info.update({
                "top_fraction": top_frac,
                "bottom_fraction": bottom_frac,
                "top_boundary_px": int(H * top_frac),
                "bottom_boundary_px": int(H * (1.0 - bottom_frac)),
                "valid_region_height_px": int(H * (1.0 - top_frac - bottom_frac))
            })

        # No detections
        if not detections:
            return {
                **self._classify_no_detections(img),
                "region_exclusion": region_info
            }

        # Separate liquid and non-liquid detections
        liquid_dets = [d for d in detections if self._is_liquid(d['class_id'])]

        # If no liquid detections, try fallback mechanisms
        if not liquid_dets:
            print("\nOP VLEZI\n")
            fallback = self._only_air_detections(
                detections=detections,
                crop_path=crop_path,
                region_info=region_info,
                H=H,
                W=W
            )
            if fallback is not None:
                return fallback

        print("\nIZLEZI\n")
        # Check for phase separation
        is_separated, phase_metrics = self._detect_phase_separation(
            liquid_dets, H, W, img, crop_path
        )

        # Check if phase separated
        if is_separated:
            return {
                "vial_state": "phase_separated",
                "phase_metrics": phase_metrics,
                "detections": self._summarize_detections(liquid_dets),
                "region_exclusion": region_info
            }
        
        # Classify as gelled or stable
        state = self._classify_liquid_state(liquid_dets, H, W, crop_path)
        
        return {
            "vial_state": state,
            "detections": self._summarize_detections(liquid_dets),
            "coverage": self._calculate_coverage(liquid_dets, H, W),
            "region_exclusion": region_info
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

        original_count = len(detections)
        print("\nOriginal detections count: ", original_count, "\n")

        # Apply region exclusion
        if detections and self.region_exclusion:
            top_frac = self.region_exclusion.get('top_fraction', 0.0)
            bottom_frac = self.region_exclusion.get('bottom_fraction', 0.0)

            if top_frac > 0 or bottom_frac > 0:
                detections, excluded = filter_detections_by_exclusion_region(
                    detections,
                    H,
                    top_frac,
                    bottom_frac,
                    return_excluded=True
                )

                # Log excluded detections
                if excluded:
                    LOGGER.info(
                        f"[{label_path.stem}] Excluded {len(excluded)}/{original_count} detections from regions")
                    LOGGER.info(
                        f"  Image height: {H}px, Top boundary: {int(H * top_frac)}px, Bottom boundary: {int(H * (1 - bottom_frac))}px")
                    for ex in excluded:
                        LOGGER.info(
                            f"    - Class {ex['class_id']} at y={ex['center_y']:.1f}px (conf={ex['confidence']:.3f})")

        # # Deduplicate overlapping detections
        # if detections and self.merge_boxes:
        #     # Define priority: AIR > GEL > STABLE
        #     priority_classes = [
        #         CLASS_IDS.get('AIR'),
        #         CLASS_IDS.get('GEL'),
        #         CLASS_IDS.get('STABLE'),
        #     ]
        #     priority_classes = [c for c in priority_classes if c is not None]
        #
        #     detections, removed = deduplicate_overlapping_detections(
        #         detections,
        #         iou_threshold=DETECTION_THRESHOLDS.get('iou_thr', 0.85),
        #         priority_classes=priority_classes
        #     )
        #
        #     # Log removed detections for debugging
        #     if removed and hasattr(self, 'debug_mode') and self.debug_mode:
        #         LOGGER.info(f"Removed {len(removed)} overlapping detections")

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

    def _only_air_detections(self, detections: List[Dict[str, Any]], crop_path: Path,
                             region_info: Dict[str, Any], H: int, W: int) -> dict[str, Any]:
        air_dets = [d for d in detections if d["class_id"] == CLASS_IDS["AIR"]]
        liquid_dets = [d for d in detections if self._is_liquid(d['class_id'])]

        if not liquid_dets and len(air_dets) == 1:
            air_box = air_dets[0]["box"]  # [x1,y1,x2,y2]
            air_h_frac = (air_box[3] - air_box[1]) / H

            # Optional: only engage fallback if AIR box is not dominating height
            if air_h_frac >= ONLY_AIR_CLASSIFICATION["min_air_height_fraction"]:
                return {
                    "vial_state": "only_air",
                    "reason": "fallback_gate:air_box_too_tall",
                    "region_exclusion": region_info,
                    "fallback": {"air_height_frac": float(air_h_frac)}
                }

            hdet = LineDetector(
                min_line_length=ONLY_AIR_CLASSIFICATION["min_horizontal_line_len_frac"],
                merge_threshold=LINE_PARAMS["merge_threshold"]
            ).detect(
                image_path=crop_path,
                top_exclusion=REGION_EXCLUSION.get("top_fraction", 0.20),
                bottom_exclusion=REGION_EXCLUSION.get("bottom_fraction", 0.10),
            )
            num_long, max_vert_gap = self._summarize_hlines(hdet)

            curve = run_curve_metrics(crop_path)
            var_val = float(curve.get("stats", {}).get("variance_from_baseline", 0.0))

            # PHASE SEPARATED: >=2 long lines & vertical gap >= threshold
            if num_long >= 2 and max_vert_gap >= ONLY_AIR_CLASSIFICATION["min_vertical_gap_frac"]:
                return {
                    "vial_state": "phase_separated",
                    "reason": "fallback_only_air:>=2_long_hlines_and_gap",
                    "fallback": {
                        "air_height_frac": float(air_h_frac),
                        "num_long_hlines": int(num_long),
                        "max_vertical_gap_frac": float(max_vert_gap),
                        "curve_variance_from_baseline": var_val,
                    },
                    "phase_metrics": {
                        "y_norm": [float(l["y_norm"]) for l in hdet["horizontal_lines"]["lines"]
                                   if float(l.get("length_frac", 0.0)) >= ONLY_AIR_CLASSIFICATION[
                                       "min_horizontal_line_len_frac"]],
                        "left_boundary_norm": float(hdet["vertical_lines"]["left_boundary"]),
                        "right_boundary_norm": float(hdet["vertical_lines"]["right_boundary"]),
                    }
                }

            # STABLE: exactly 1 long line & curve looks stable
            if num_long == 1 and self._curve_is_stable(curve):
                return {
                    "vial_state": "stable",
                    "reason": "fallback_only_air:1_long_hline_and_curve_stable",
                    "fallback": {
                        "air_height_frac": float(air_h_frac),
                        "num_long_hlines": int(num_long),
                        "max_vertical_gap_frac": float(max_vert_gap),
                        "curve_variance_from_baseline": var_val,
                    }
                }

            # GELLED: 0 or 1 long line & curve looks gelled
            if num_long <= 1 and self._curve_is_gelled(curve):
                return {
                    "vial_state": "gelled",
                    "reason": "fallback_only_air:<=1_long_hline_and_curve_gelled",
                    "fallback": {
                        "air_height_frac": float(air_h_frac),
                        "num_long_hlines": int(num_long),
                        "max_vertical_gap_frac": float(max_vert_gap),
                        "curve_variance_from_baseline": var_val,
                    }
                }

            # No decisive outcome
            return {
                "vial_state": "only_air",
                "reason": "fallback_only_air:no_decision",
                "fallback": {
                    "air_height_frac": float(air_h_frac),
                    "num_long_hlines": int(num_long),
                    "max_vertical_gap_frac": float(max_vert_gap),
                    "curve_variance_from_baseline": var_val,
                }
            }

        # Default path if conditions not met
        return {
            "vial_state": "only_air",
            "reason": "no_liquid_detected",
            "total_detections": len(detections),
            "region_exclusion": region_info
        }


    def _detect_phase_separation(self, liquid_dets: List[Dict[str, Any]],
                                 H: int, W: int, 
                                 img: Optional[np.ndarray] = None,
                                 img_path: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if vial has phase separation.
        
        Returns:
            Tuple of (is_separated, metrics)
        """
        # Line detection
        if self.use_line_detection:
            # instantiate class
            ps = PhaseSeparationDetector()

            # detect lines
            is_separated, line_metrics = ps.detect(
                detections=liquid_dets,
                image_height=H,
                image_width=W,
                cap_bottom_y=None,
                image_path=img_path
            )

            # print number of lines detected
            if 'num_horizontal_lines' in line_metrics:
                print(f"\nNumber of lines detected: {line_metrics['num_horizontal_lines']}")

            if is_separated:
                    return True, {'method': 'line', **line_metrics}

        # Turbidity analysis (if enabled and image available)
        # if self.use_turbidity and img is not None:
        #     profile = compute_turbidity_profile(img)
        #     is_separated, turbidity_metrics = detect_turbidity_peaks(profile)
        #
        #     if is_separated:
        #         return True, {'method': 'turbidity', **turbidity_metrics}
        
        return False, {'method': 'none'}

    def _count_horizontal_lines(self, crop_path: Path, H: int) -> Dict[str, Any]:
        """
        Count horizontal lines in a cropped image.
        Args:
            crop_path: path to cropped image
            H: height of cropped image

        Returns:
            Dictionary with count of horizontal lines and their statistics.
        """
        # init detector
        detector = LineDetector(min_line_length=LINE_PARAMS["min_line_length"])
        # parse results
        results = detector.detect(
            crop_path,
            top_exclusion=REGION_EXCLUSION.get("top_fraction", 0.20),
            bottom_exclusion=REGION_EXCLUSION.get("bottom_fraction", 0.10)
        )
        # horizontal lines info
        hlines = results["horizontal_lines"]["lines"]

        return {
            "num_horizontal_lines": int(results["horizontal_lines"]["num_lines"]),
            "y_norm": [float(l["y_norm"]) for l in hlines],
            "y_px": [int(round(float(l["y_norm"]) * H)) for l in hlines],
            "boundaries": (
                float(results["vertical_lines"]["left_boundary"]),
                float(results["vertical_lines"]["right_boundary"])
            )
        }


    def _summarize_hlines(self, hdet: dict) -> tuple[int, float]:
        """
        Returns (num_long_lines, max_vertical_gap_frac) using normalized values.
        """
        # Get horizontal lines
        lines = hdet.get("horizontal_lines", {}).get("lines", [])

        # Extract only lines of valid length
        valid = [l for l in lines if float(l.get("length_frac", 0.0))
                 >= ONLY_AIR_CLASSIFICATION["min_horizontal_line_len_frac"]]

        # Sort lines by y-pos and find max vertical gap
        ys = sorted(float(l.get("y_norm")) for l in valid)
        max_gap = 0.0
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                max_gap = max(max_gap, ys[j] - ys[i])
        return len(valid), max_gap


    def _curve_is_stable(self, curve: dict) -> bool:
        """
        Check if curve is characteristic for stable state.
        """
        var_ = float(curve.get("stats", {}).get("variance_from_baseline", float("inf")))
        return var_ < CURVE_PARAMS.get("stable_variance_thr", 50.0)


    def _curve_is_gelled(self, curve: dict) -> bool:
        """
        Check if curve is characteristic for gel state.
        """
        var_ = float(curve.get("stats", {}).get("variance_from_baseline", -1.0))
        return var_ >= CURVE_PARAMS.get("gel_variance_thr", 80.0)


    def _classify_liquid_state(self, liquid_dets: List[Dict[str, Any]],
                               H: int, W: int, crop_path: Path) -> str:
        """
        Classify liquid state based on detected gel lines.

        Returns:
            'gelled' if gel-like lines are detected, 'stable' otherwise
        """
        if self.use_curved_line_detection:
            try:
                if run_curve_metrics(crop_path)['gelled_by_curve']:
                    return 'gelled'
            except Exception as e:
                return f"reason: curve error: {str(e)}"

        if self._count_horizontal_lines(crop_path, H).get('num_horizontal_lines') == 1:
            return 'stable'

        total_area = sum(d['area'] for d in liquid_dets)
        gel_area = sum(d['area'] for d in liquid_dets if d['class_id'] == CLASS_IDS['GEL'])
        n_gel = sum(1 for d in liquid_dets if d['class_id'] == CLASS_IDS['GEL'])
        n_stable = sum(1 for d in liquid_dets if d['class_id'] == CLASS_IDS['STABLE'])

        if total_area > 0 and gel_area / total_area >= DETECTION_THRESHOLDS['gel_area_frac']:
            return 'gelled'
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