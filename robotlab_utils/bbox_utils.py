# robotlab_utils/bbox_utils.py
"""
Bounding box utility functions.
Provides helpers for converting formats, expanding/clamping,
computing box metrics, and filtering detections.
"""

from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class Letterbox:
    gain: float       # scale factor applied during preprocess
    pad: tuple[float, float]  # (pad_x, pad_y)

def yolo_line_to_xyxy_px(line: str, W: int, H: int, normalized: bool=True):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0]); cx, cy, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    if normalized:
        cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    return cls, [x1, y1, x2, y2], conf

def undo_letterbox_xyxy(xyxy_net, lb: Letterbox):
    x1,y1,x2,y2 = map(float, xyxy_net)
    px,py = lb.pad
    g = lb.gain if lb.gain != 0 else 1.0
    return [(x1 - px)/g, (y1 - py)/g, (x2 - px)/g, (y2 - py)/g]

def to_crop_space_xyxy(xyxy_src, crop_xyxy):
    # crop_xyxy = [x0,y0,x1,y1] in source px; top-left offset is (x0,y0)
    x0, y0 = float(crop_xyxy[0]), float(crop_xyxy[1])
    x1,y1,x2,y2 = map(float, xyxy_src)
    return [x1 - x0, y1 - y0, x2 - x0, y2 - y0]

def scale_xyxy(xyxy, scale):
    return [v * float(scale) for v in xyxy]

def clamp_order_xyxy(xyxy, W: int, H: int):
    x1, y1, x2, y2 = map(float, xyxy)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W - 1.0, x2))
    y2 = max(0.0, min(H - 1.0, y2))
    return [x1, y1, x2, y2]

def ensure_xyxy_px(det: dict, W: int, H: int):
    """
    Accepts either:
      - {'box': [x1,y1,x2,y2], 'confidence': ..., ...}  (pixel space)
      - or YOLO-normalized {'cx','cy','w','h'} (fractions of the crop).
    Returns pixel xyxy, clamped and ordered.
    """
    if 'box' in det and det['box'] is not None:
        return clamp_order_xyxy(det['box'], W, H)

    # fall back to normalized xywh → pixel xyxy
    cx, cy, w, h = det['cx'], det['cy'], det['w'], det['h']
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return clamp_order_xyxy([x1, y1, x2, y2], W, H)

def assert_box(xyxy, W, H, tag=""):
    x1,y1,x2,y2 = xyxy
    ok = 0 <= x1 <= x2 < W and 0 <= y1 <= y2 < H
    if not ok:
        print(f"[BAD BOX {tag}] {xyxy} in ({W}x{H})")
    return ok


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
    box = clamp_order_xyxy([x1, y1, x2, y2], W, H)
    assert_box(box, W, H, tag="yolo conversion")
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


# def iou_xyxy(a: List[float], b: List[float]) -> float:
#     """
#     Calculate Intersection over Union (IoU) between two bounding boxes.
#
#     Args:
#         a, b: [x1, y1, x2, y2]
#
#     Returns:
#         IoU value between 0 and 1
#     """
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#     ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
#     inter = iw * ih
#     ua = max(0, ax2-ax1) * max(0, ay2-ay1) + max(0, bx2-bx1) * max(0, by2-by1) - inter
#     # print("\niou_xyxy: ", ua, "\n")
#     return inter / (ua + 1e-9)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    print("\ncompute iou: ", inter_area / union_area, "\n")

    return inter_area / union_area if union_area > 0 else 0.0


def merge_detections_by_iou(
        dets: List[Dict],
        iou_thr: float = 0.5
) -> List[Dict]:
    """
    Greedy merge: sort by confidence descending, keep a detection
    only if it doesn't overlap too much (IoU > threshold) with any
    already kept detection.

    Args:
        dets: List of detection dicts with 'box', 'confidence'
        iou_thr: IoU threshold - boxes with IoU > this are considered overlapping

    Returns:
        List of non-overlapping detections (highest confidence kept)
    """
    if not dets:
        return dets

    # Sort by confidence (highest first)
    dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)

    kept = []
    for d in dets:
        # Check if this box overlaps with any already kept box
        overlaps_with_kept = any(
            compute_iou(d['box'], k['box']) > iou_thr
            for k in kept
        )

        # Only keep if it doesn't overlap
        if not overlaps_with_kept:
            kept.append(d)

    return kept


def is_detection_in_excluded_region(
        detection: Dict[str, Any],
        image_height: int,
        top_fraction: float = 0.0,
        bottom_fraction: float = 0.0
    ) -> bool:
    """
    Check if detection center is in excluded region.

    Args:
        detection: Detection dictionary with 'center_y' or 'box' key
        image_height: Height of image in pixels
        top_fraction: Fraction of image height to exclude from top (0.0-1.0)
        bottom_fraction: Fraction of image height to exclude from bottom (0.0-1.0)

    Returns:
        True if detection is in excluded region, False otherwise
    """
    # Get detection center Y coordinate
    if 'center_y' in detection:
        center_y = detection['center_y']
    elif 'box' in detection:
        box = detection['box']
        center_y = (box[1] + box[3]) / 2.0
    else:
        return False

    # Calculate exclusion boundaries
    top_boundary = image_height * top_fraction
    bottom_boundary = image_height * (1.0 - bottom_fraction)

    # Check if center is in excluded region
    if center_y < top_boundary or center_y > bottom_boundary:
        return True

    return False


def filter_detections_by_exclusion_region(
        detections: List[Dict[str, Any]],
        image_height: int,
        top_fraction: float = 0.0,
        bottom_fraction: float = 0.0,
        return_excluded: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Filter detections by excluding top and bottom regions.

    Args:
        detections: List of detection dictionaries
        image_height: Height of image in pixels
        top_fraction: Fraction to exclude from top (0.0-1.0)
        bottom_fraction: Fraction to exclude from bottom (0.0-1.0)
        return_excluded: If True, return both kept and excluded detections

    Returns:
        Filtered detections list, or (kept, excluded) tuple if return_excluded=True
    """
    if not detections or (top_fraction == 0.0 and bottom_fraction == 0.0):
        return (detections, []) if return_excluded else detections

    kept = []
    excluded = []

    for det in detections:
        if is_detection_in_excluded_region(det, image_height, top_fraction, bottom_fraction):
            excluded.append({
                **det,
                'exclusion_reason': 'region_excluded'
            })
        else:
            kept.append(det)

    if return_excluded:
        return kept, excluded
    return kept


# --- cross-class conflict resolution ---
# def deduplicate_overlapping_detections(
#         detections: List[Dict[str, Any]],
#         iou_threshold: float = 0.85,
#         priority_classes: Optional[List[int]] = None
# ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     """
#     Remove overlapping detections, keeping the most dominant one.
#
#     Args:
#         detections: List of detection dictionaries
#         iou_threshold: IoU threshold for considering boxes as overlapping
#         priority_classes: Optional list of class IDs in priority order (highest first)
#
#     Returns:
#         Tuple of (kept_detections, removed_detections)
#     """
#     if not detections:
#         return [], []
#
#     kept = []
#     removed = []
#     used_indices = set()
#
#     # Sort by confidence (descending)
#     sorted_dets = sorted(enumerate(detections),
#                          key=lambda x: x[1]['confidence'],
#                          reverse=True)
#
#     for idx, det in sorted_dets:
#         if idx in used_indices:
#             continue
#
#         # Find all detections that overlap with this one
#         overlapping = []
#         for other_idx, other_det in sorted_dets:
#             if other_idx == idx or other_idx in used_indices:
#                 continue
#
#             iou = compute_iou(det['box'], other_det['box'])
#             if iou >= iou_threshold:
#                 overlapping.append((other_idx, other_det, iou))
#
#         if overlapping:
#             # Add current detection to candidates
#             candidates = [(idx, det, 1.0)] + overlapping
#
#             # Select best detection based on priority
#             best_det = select_dominant_detection(
#                 candidates,
#                 priority_classes
#             )
#
#             kept.append(best_det[1])
#
#             # Mark all overlapping as used
#             used_indices.add(idx)
#             for overlap_idx, overlap_det, _ in overlapping:
#                 used_indices.add(overlap_idx)
#                 removed.append({
#                     **overlap_det,
#                     'removed_reason': 'overlapping',
#                     'overlaps_with': best_det[1]['class_id'],
#                     'iou': overlap_idx
#                 })
#         else:
#             kept.append(det)
#             used_indices.add(idx)
#
#     return kept, removed
#
#
# def select_dominant_detection(
#         candidates: List[Tuple[int, Dict[str, Any], float]],
#         priority_classes: Optional[List[int]] = None
# ) -> Tuple[int, Dict[str, Any], float]:
#     """
#     Select the dominant detection from overlapping candidates.
#
#     Args:
#         candidates: List of (index, detection, iou) tuples
#         priority_classes: Optional class priority list
#
#     Returns:
#         The dominant (index, detection, iou) tuple
#     """
#     if len(candidates) == 1:
#         return candidates[0]
#
#     # Strategy 1: Use class priority if provided
#     if priority_classes:
#         for priority_class in priority_classes:
#             for candidate in candidates:
#                 if candidate[1]['class_id'] == priority_class:
#                     return candidate
#
#     # Strategy 2: Highest confidence
#     return max(candidates, key=lambda x: x[1]['confidence'])

# def merge_detections_by_iou_with_priority(
#     dets,
#     iou_thr=0.5,
#     *,
#     image_height=None,
#     air_class_id=2,
#     top_region_fraction=0.20,
#     conflict_iou=0.8
# ):
#     """
#     Single-call cleanup:
#       (0) prioritize AIR in the top region (domain rule),
#       (1) suppress cross-class conflicts (keep highest conf),
#       (2) same-class greedy NMS (keep highest conf within each class).
#     """
#     if not dets:
#         return dets
#
#     # Compute top region cutoff using the real image height, not box coords
#     if image_height is None:
#         image_height = max(int(d['box'][3]) for d in dets)  # fallback
#
#     top_y = image_height * top_region_fraction
#
#     # Split into AIR-top and the rest
#     air_top, others = [], []
#     for d in dets:
#         y1 = d['box'][1]
#         (air_top if (d['class_id'] == air_class_id and y1 < top_y) else others).append(d)
#
#     # keep AIR in top region with same-class NMS
#     air_top = _merge_same_class_by_iou(air_top, iou_thr=iou_thr)
#
#     # Combine with others and suppress cross-class conflicts globally
#     combined = air_top + others
#     combined = _suppress_conflicting_detections(combined, iou_thr=conflict_iou)
#
#     # same-class NMS on the result
#     cleaned = _merge_same_class_by_iou(combined, iou_thr=iou_thr)
#     return cleaned


# def filter_detections_by_region(
#     dets: List[Dict], region: Tuple[int, int, int, int]
# ) -> List[Dict]:
#     """
#     Keep only detections whose centers fall inside a region.
#
#     Args:
#         dets: List of detection dicts with key 'box' = [x1, y1, x2, y2]
#         region: (rx1, ry1, rx2, ry2)
#
#     Returns:
#         Filtered detections
#     """
#     rx1, ry1, rx2, ry2 = region
#     out = []
#     for d in dets:
#         x1, y1, x2, y2 = d['box']
#         cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#         if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
#             out.append(d)
#     return out
