#!/bin/bash
# Test script for gel line detection
# Usage: ./test_gel_detection.sh <path_to_image>

echo "=========================================="
echo "Gel Line Detection Test Script"
echo "=========================================="
echo ""

# Check if image path provided
if [ -z "$1" ]; then
    echo "Error: Please provide an image path"
    echo "Usage: ./test_gel_detection.sh <path_to_image>"
    exit 1
fi

IMAGE_PATH="$1"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image not found: $IMAGE_PATH"
    exit 1
fi

echo "Testing gel detection on: $IMAGE_PATH"
echo ""

# Create horizontal_line_detection_results directory
OUTPUT_DIR="gel_detection_output"
mkdir -p "$OUTPUT_DIR"

# Extract filename without extension
FILENAME=$(basename "$IMAGE_PATH")
STEM="${FILENAME%.*}"

# Run detection with different parameter sets
echo "Running detection with default parameters..."
python3 gel_line_detector.py "$IMAGE_PATH" \
    --output "$OUTPUT_DIR/${STEM}_default.png" \
    --json "$OUTPUT_DIR/${STEM}_default.json" \
    --save-intermediates \
    --verbose

echo ""
echo "----------------------------------------"
echo ""

echo "Running detection with sensitive parameters (more likely to detect gel)..."
python3 gel_line_detector.py "$IMAGE_PATH" \
    --output "$OUTPUT_DIR/${STEM}_sensitive.png" \
    --json "$OUTPUT_DIR/${STEM}_sensitive.json" \
    --min-gel-lines 2 \
    --min-sinuous 0.10 \
    --min-discontinuity 0.3 \
    --min-x-span 0.15 \
    --verbose

echo ""
echo "----------------------------------------"
echo ""

echo "Running detection with strict parameters (less likely to detect gel)..."
python3 gel_line_detector.py "$IMAGE_PATH" \
    --output "$OUTPUT_DIR/${STEM}_strict.png" \
    --json "$OUTPUT_DIR/${STEM}_strict.json" \
    --min-gel-lines 2 \
    --min-sinuous 0.30 \
    --min-discontinuity 0.5 \
    --min-x-span 0.35 \
    --verbose

echo ""
echo "=========================================="
echo "Testing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo "=========================================="