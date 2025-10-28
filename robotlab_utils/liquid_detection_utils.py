"""
Utility functions for parsing YOLO detections.
"""

from pathlib import Path
from typing import List, Dict, Any
import sys

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robotlab_utils.bbox_utils import yolo_line_to_xyxy, box_area
from config import DETECTION_FILTERS, LIQUID_CLASSES


def parse_detections(label_path: Path, W: int, H: int) -> List[Dict[str, Any]]:
    """
    Parse YOLO label file into detection dictionaries, with filtering.

    Args:
        label_path: Path to YOLO label file
        W: Image width
        H: Image height

    Returns:
        List of detection dictionaries
    """
    detections = []

    if label_path.exists():
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
                if conf < DETECTION_FILTERS["conf_min"]:
                    continue

                area = box_area(box)

                # Filter small liquid areas
                if area < DETECTION_FILTERS["min_liquid_area_frac"] * (W * H) and cls_id in LIQUID_CLASSES:
                    continue

                det = {
                    'class_id': cls_id,
                    'box': box,
                    'confidence': conf,
                    'area': area
                }
                detections.append(det)

    return detections