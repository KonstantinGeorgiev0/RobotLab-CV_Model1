import cv2
import numpy as np
from pathlib import Path
from config import CLASS_IDS
from robotlab_utils.bbox_utils import yolo_line_to_xyxy


def create_filtered_detection_visualization(
        crop_path: Path,
        label_path: Path,
        output_path: Path,
        top_fraction: float = 0.0,
        bottom_fraction: float = 0.0,
        show_excluded_regions: bool = True
    ) -> Path:
    """
    Create visualization with only non-excluded detections.

    Args:
        crop_path: Path to crop image
        label_path: Path to label file
        output_path: Where to save visualization
        top_fraction: Top exclusion fraction
        bottom_fraction: Bottom exclusion fraction
        show_excluded_regions: Draw red overlay on excluded regions

    Returns:
        Path to saved visualization
    """

    # Load image
    img = cv2.imread(str(crop_path))
    if img is None:
        return output_path

    H, W = img.shape[:2]
    vis_img = img.copy()

    # Calculate boundaries
    top_boundary = int(H * top_fraction)
    bottom_boundary = int(H * (1.0 - bottom_fraction))

    # Draw excluded regions if requested
    if show_excluded_regions:
        overlay = vis_img.copy()

        # Top region
        if top_fraction > 0:
            cv2.rectangle(overlay, (0, 0), (W, top_boundary), (0, 0, 255), -1)

        # Bottom region
        if bottom_fraction > 0:
            cv2.rectangle(overlay, (0, bottom_boundary), (W, H), (0, 0, 255), -1)

        # Blend
        vis_img = cv2.addWeighted(overlay, 0.2, vis_img, 0.8, 0)

        # Draw boundary lines
        if top_fraction > 0:
            cv2.line(vis_img, (0, top_boundary), (W, top_boundary), (0, 0, 255), 2)
            cv2.putText(vis_img, f"Top exclusion: {top_fraction * 100:.0f}%",
                        (10, top_boundary - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        if bottom_fraction > 0:
            cv2.line(vis_img, (0, bottom_boundary), (W, bottom_boundary), (0, 0, 255), 2)
            cv2.putText(vis_img, f"Bottom exclusion: {bottom_fraction * 100:.0f}%",
                        (10, bottom_boundary + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    # Parse and draw only non-excluded detections
    if not label_path.exists():
        cv2.imwrite(str(output_path), vis_img)
        return output_path

    kept_count = 0
    excluded_count = 0

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            parsed = yolo_line_to_xyxy(line.strip(), W, H)
            if not parsed:
                continue

            cls_id, box, conf = parsed
            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) / 2.0

            # Check if in excluded region
            is_excluded = center_y < top_boundary or center_y > bottom_boundary

            if is_excluded:
                excluded_count += 1
                continue

            # Draw detection
            kept_count += 1

            # Get class name
            class_name = CLASS_IDS.get(cls_id, f"Class_{cls_id}")

            # Color based on class
            if cls_id == CLASS_IDS.get('GEL'):
                color = (0, 165, 255)  # Orange
            elif cls_id == CLASS_IDS.get('STABLE'):
                color = (0, 255, 0)  # Green
            elif cls_id == CLASS_IDS.get('AIR'):
                color = (255, 0, 0)  # Blue
            else:
                color = (128, 128, 128)  # Gray

            # Draw box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 4),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add summary text
    summary = f"Kept: {kept_count} | Excluded: {excluded_count}"
    cv2.putText(vis_img, summary, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)

    return output_path