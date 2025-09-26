# Vial & Liquid Detection Pipeline

## Overview

This pipeline provides a modular, maintainable solution for detecting vials in images and analyzing their liquid content. The pipeline consists of two main stages:

1. **Vial Detection**: Detects vials in images and creates cropped images
2. **Liquid Analysis**: Analyzes liquid content and classifies vial states

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
vial_detection_pipeline/
├── main.py                      # Main pipeline orchestration
├── config.py                    # Configuration and constants
├── detection/                   # Detection modules
│   ├── vial_detector.py
│   └── liquid_detector.py
├── analysis/                    # Analysis modules
│   ├── phase_separation.py
│   ├── state_classifier.py
│   └── turbidity_analysis.py
├── utils/                       # Utility functions
│   ├── bbox_utils.py
│   ├── image_utils.py
│   └── io_utils.py
└── visualization/               # Visualization modules
    └── turbidity_viz.py
```

## Usage

### Basic Usage

```bash
python main.py \
    --source path/to/images \
    --vial-weights models/vial_detector.pt \
    --liquid-weights models/liquid_detector.pt \
    --outdir results/
```

### Advanced Options

```bash
python main.py \
    --source path/to/images \
    --vial-weights path/to/models/vial_detector.pt \
    --liquid-weights path/to/models/liquid_detector.pt \
    --outdir results/ \
    --vial-conf 0.7 \
    --liquid-conf 0.3 \
    --use-turbidity \
    --merge-boxes \
    --save-turbidity \
```

### Command-line Arguments

#### Required Arguments
- `--source`: Input images/directory/video
- `--vial-weights`: Path to vial detection model
- `--liquid-weights`: Path to liquid detection model

#### Optional Arguments

**Output:**
- `--outdir`: Output directory (default: `runs/vial2liquid`)

**Vial Detection (Stage A):**
- `--imgsz`: Inference size (default: 640)
- `--vial-conf`: Confidence threshold (default: 0.65)
- `--vial-iou`: IoU threshold (default: 0.45)
- `--pad`: Padding fraction for crops (default: 0.12)
- `--crop-h`: Target height for crops (default: 640)
- `--topk`: Keep top-K vials per image (default: -1 for all)

**Liquid Detection (Stage B):**
- `--liquid-task`: Task type: "detect" or "segment" (default: detect)
- `--liquid-imgsz`: Image size (default: 640)
- `--liquid-conf`: Confidence threshold (default: 0.25)
- `--liquid-iou`: IoU threshold (default: 0.50)

**Analysis Options:**
- `--use-turbidity`: Enable turbidity analysis for phase separation
- `--merge-boxes`: Merge overlapping detection boxes
- `--save-turbidity`: Save turbidity plots

**Hardware:**
- `--device`: CUDA device (default: auto-select)
- `--half`: Use FP16 inference

## Output Structure

```
results/
├── crops/                       # Cropped vial images
│   └── exp/
├── liquid_runs/                 # Liquid detection results
│   └── exp/
│       ├── labels/             # YOLO format labels
│       ├── manifest/           # JSON manifest files
│       ├── turbidity/          # Turbidity plots
│       └── visualizations/     # Detection visualizations
```

## Vial State Classification

The pipeline classifies vials into four states:

1. **Stable**: Liquid without gel formation
2. **Gelled**: Liquid with gel formation
3. **Phase Separated**: Multiple distinct liquid layers
4. **Only Air**: No liquid detected

## Configuration

Edit `config.py` to adjust default parameters:

```python
# Example: Adjust detection thresholds
DETECTION_THRESHOLDS = {
    'conf_min': 0.20,
    'iou_merge': 0.50,
    'gel_area_frac': 0.35,
    'gel_dominance_count': 1,
}

# Example: Adjust turbidity parameters
TURBIDITY_PARAMS = {
    'gradient_threshold_sigma': 2.5,
    'gradient_threshold_min': 0.10,
    ...
}
```

## Manifest File Format

The pipeline generates a JSONL manifest file with detailed information:

```json
{
  "source_image": "path/to/original.jpg",
  "crop_image": "path/to/crop.png",
  "vial_index": 0,
  "det_conf": 0.95,
  "vial_state": "stable",
  "detections": {
    "n_gel": 0,
    "n_stable": 2,
    "gel_area_fraction": 0.0
  },
  "turbidity_stats": {
    "mean": 0.45,
    "std": 0.12,
    "max_gradient": 0.08
  }
}
```

## Troubleshooting

### Common Issues

1. **No detections found**
   - Lower confidence thresholds
   - Check model compatibility
   - Verify image format

2. **Memory issues**
   - Use `--half` for FP16 inference
   - Reduce batch size
   - Process images in smaller batches

3. **Incorrect state classification**
   - Adjust thresholds in `config.py`
   - Enable `--use-turbidity` for better phase separation detection
   - Use `--merge-boxes` to reduce duplicate detections


## Contributing

When adding new features:

1. Add new modules in appropriate directories
2. Update configuration in `config.py`
3. Maintain consistent coding style
4. Add comprehensive docstrings
5. Update this README

