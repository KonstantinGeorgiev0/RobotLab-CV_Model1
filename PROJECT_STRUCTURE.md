# RobotLab-CV_Model1 Project Structure

This document explains the purpose of each key file, the methods inside them, their inputs/outputs, and how they connect in the pipeline.

---

## **main.py**
**Purpose:** Orchestrates the entire detection pipeline:
1. Run vial detection (Stage A).
2. Run liquid detection on vial crops (Stage B).
3. Classify vial states (Stage C).
4. Perform turbidity/gradient analysis.
5. Save outputs (plots + manifest).

### Key Components
- `VialDetectionPipeline`: main pipeline class.
  - **Inputs:** command-line args (source image/video, weights, options).
  - **Outputs:** crops, liquid detections, state classifications, turbidity plots, manifest JSON.
- `run()`: executes stages A–C in sequence.
- `process_results()`: parses liquid detection results, calls state classifier and turbidity analysis.
- `generate_summary()`: aggregates statistics across vials.

---

## **detection/vial_detector.py**
**Purpose:** Detect vials in the source image (Stage A).

### Methods
- `VialDetector.detect_and_crop(img_path)`
  - **Inputs:** path to an image, YOLO vial weights.
  - **Process:** runs YOLOv5 directly (DetectMultiBackend), finds vial bounding boxes, crops them.
  - **Outputs:** cropped vial images (`results/crops/exp*/...`), list of crop records (paths + bbox metadata).

---

## **detection/liquid_detector.py**
**Purpose:** Detect liquid/air layers inside vial crops (Stage B).

### Methods
- `LiquidDetector.detect(crop_dir, out_dir)`
  - **Inputs:** directory of vial crops, YOLO liquid weights.
  - **Process:** calls `yolov5/detect.py` via subprocess to run inference.
  - **Outputs:** YOLO label files in `results/liquid_runs/exp*/labels/`, detection metadata for classifier.

---

## **analysis/state_classifier.py**
**Purpose:** Interpret YOLO liquid detections and classify each vial as **stable / gelled / phase-separated**.

### Methods
- `classify(crop_path, label_path)`
  - **Inputs:** vial crop path, YOLO label file path.
  - **Process:** parse YOLO labels → convert to pixel boxes → apply rules:
    - Multiple liquid regions → likely phase separation.
    - Vertical gap or span → phase separation.
    - High brightness or texture → gel.
    - Else stable.
  - **Outputs:** classification dict with `state`, `confidence`, optional viz/turbidity paths.
- `_parse_detections(label_path, W, H)`
  - **Inputs:** YOLO label file, image dimensions.
  - **Outputs:** list of parsed detections in xyxy format.

---

## **analysis/turbidity_analysis.py**
**Purpose:** Compute turbidity profiles and gradient-based phase-separation cues.

### Methods
- `compute_turbidity_profile(img)`
  - **Inputs:** cropped vial image.
  - **Process:** convert to grayscale → compute row-wise brightness → normalize to [0,1].
  - **Outputs:** `TurbidityProfile` (normalized profile + metadata).
- `detect_turbidity_peaks(profile)`
  - **Inputs:** turbidity profile.
  - **Process:** compute gradient, threshold, detect peaks, group them.
  - **Outputs:** `(is_separated, metrics_dict)`.

---

## **analysis/phase_separation.py**
**Purpose:** Implements decision logic for phase separation using turbidity + gradient.

### Methods
- `enhanced_detect_phase_separation_from_turbidity(img, cap_bottom_y)`
  - **Inputs:** vial crop, optional cap bottom.
  - **Outputs:** `True/False` (phase separation), diagnostic metrics.

---

## **visualization/turbidity_viz.py**
**Purpose:** Generate plots for turbidity and gradient analysis.

### Methods
- `save_turbidity_plot(path, v_norm, out_dir)`
  - **Inputs:** image path, normalized turbidity vector.
  - **Outputs:** `.png` plot of turbidity profile.
- `save_enhanced_turbidity_plot(path, v_norm, excluded_info, out_dir)`
  - **Inputs:** turbidity profile, exclusion info.
  - **Outputs:** `.png` with **two subplots**: turbidity profile + gradient analysis with peaks.

---

## **robotlab_utils/bbox_utils.py**
**Purpose:** Bounding box helpers for vial & liquid detection.

### Methods
- `expand_and_clamp(x1,y1,x2,y2,W,H,pad_frac)`  
  Expand box with padding, clamp to image.
- `yolo_to_xyxy(cx,cy,w,h,img_w,img_h)`  
  Convert YOLO normalized → pixel xyxy.
- `yolo_line_to_xyxy(line, W, H)`  
  Parse YOLO label line → `(class_id, [x1,y1,x2,y2], conf)`.
- `box_area(bbox)`  
  Compute box area.
- `iou_xyxy(a,b)`  
  Compute IoU.
- `merge_detections_by_iou(dets,iou_thr)`  
  Greedy merge by IoU threshold.
- `filter_detections_by_region(dets, region)`  
  Keep only detections inside a region.

---

## **robotlab_utils/image_utils.py**
**Purpose:** Image preprocessing helpers.

### Methods
- `resize_keep_height(img, target_h)`  
  Resize with aspect ratio preserved.
- `extract_region(img, bbox, min_size=10)`  
  Crop ROI from bbox.
- `calculate_brightness_stats(img)`  
  Compute brightness mean/std/min/max/median.
- `apply_adaptive_threshold(img, block_size, C)`  
  Adaptive thresholding.
- `enhance_contrast(img, clip_limit, tile_size)`  
  CLAHE contrast enhancement.

---

## **robotlab_utils/io_utils.py**
**Purpose:** File I/O utilities for labels and manifests.

### Methods
- `read_jsonl(path)` / `write_jsonl(data,path)` / `append_jsonl(record,path)`  
  Read/write JSON Lines manifests.
- `read_yolo_labels(label_path)`  
  Parse YOLO label files → list of detections (`class_id, cx, cy, w, h, conf`).

---

## **Connections**

1. `main.py` orchestrates the pipeline.  
2. Stage A → `vial_detector.py` produces crops.  
3. Stage B → `liquid_detector.py` calls YOLO detect on crops.  
4. Stage C → `state_classifier.py` interprets YOLO label files.  
   - Uses `bbox_utils` for geometry.
   - Uses `turbidity_analysis` for profile + gradient.  
   - Uses `turbidity_viz` if `--save-turbidity`.  
5. `io_utils` manages manifest (JSONL).  
6. `image_utils` is used in multiple stages for preprocessing.  

---
