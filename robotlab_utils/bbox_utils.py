# robotlab_utils/bbox_utils.py
"""
Bounding box utility functions.
Provides helpers for converting formats, expanding/clamping,
computing box metrics, and filtering detections.
"""

from typing import List, Tuple, Dict


def expand_and_clamp(
    x1: int, y1: int, x2: int, y2: int,
    W: int, H: int, pad_frac: float
) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by a padding fraction and clamp it to image bounds.

    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        W, H: Image width and height
        pad_frac: Fraction to expand bbox by (relative to width and height)

    Returns:
        Expanded and clamped bounding box (x1n, y1n, x2n, y2n)
    """
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    w2, h2 = w * (1 + pad_frac), h * (1 + pad_frac)

    x1n = max(int(round(cx - w2 / 2)), 0)
    y1n = max(int(round(cy - h2 / 2)), 0)
    x2n = min(int(round(cx + w2 / 2)), W - 1)
    y2n = min(int(round(cy + h2 / 2)), H - 1)

    return x1n, y1n, x2n, y2n


def yolo_line_to_xyxy(line: str, W: int, H: int):
    """
    Converts a YOLO format line into (class_id, [x1,y1,x2,y2], confidence).

    Args:
        line: YOLO label line ("cls cx cy w h [conf]")
        W, H: Image width and height

    Returns:
        Tuple (cls_id, [x1, y1, x2, y2], conf)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    return cls, [x1, y1, x2, y2], conf


def box_area(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Area of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: List[float], b: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        a, b: [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter

    return inter / (ua + 1e-9)


def merge_detections_by_iou(
    dets: List[Dict], iou_thr: float = 0.5
) -> List[Dict]:
    """
    Greedy merge: sort by confidence, keep a detection only if it
    doesn’t overlap (IoU > thr) with already kept detections.

    Args:
        dets: List of detection dicts with keys 'box' and 'confidence'
        iou_thr: IoU threshold

    Returns:
        List of merged detections
    """
    if not dets:
        return dets

    dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
    kept = []
    for d in dets:
        bb = d['box']
        if all(iou_xyxy(bb, k['box']) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def filter_detections_by_region(
    dets: List[Dict], region: Tuple[int, int, int, int]
) -> List[Dict]:
    """
    Keep only detections whose centers fall inside a region.

    Args:
        dets: List of detection dicts with key 'box' = [x1, y1, x2, y2]
        region: (rx1, ry1, rx2, ry2)

    Returns:
        Filtered detections
    """
    rx1, ry1, rx2, ry2 = region
    out = []
    for d in dets:
        x1, y1, x2, y2 = d['box']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            out.append(d)
    return out
