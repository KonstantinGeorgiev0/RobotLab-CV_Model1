#!/usr/bin/env python3
"""
Main pipeline for vial detection and liquid analysis.
Orchestrates the two-stage process:
1. Detect vials and create crops
2. Analyze liquid content in crops
"""

import sys
from pathlib import Path

# Ensure yolov5 is on sys.path so "from utils import ..." resolves to yolov5/utils
YOLO_ROOT = Path(__file__).parent / "yolov5"
sys.path.insert(0, str(YOLO_ROOT))

import argparse
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch

# YOLOv5 imports
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    non_max_suppression, scale_boxes, check_img_size, 
    LOGGER, increment_path
)
from yolov5.utils.torch_utils import select_device

# Custom modules
from config import DEFAULT_PATHS
from analysis.state_classifier import VialStateClassifier
from analysis.turbidity_analysis import compute_turbidity_profile, analyze_region_turbidity
from visualization.turbidity_viz import save_enhanced_turbidity_plot
from robotlab_utils.image_utils import resize_keep_height
from robotlab_utils.bbox_utils import expand_and_clamp


class VialDetectionPipeline:
    """Main pipeline for vial detection and analysis."""
    
    def __init__(self, args):
        """Initialize pipeline with command-line arguments."""
        self.args = args
        self.device = select_device(args.device)
        self.classifier = VialStateClassifier(
            use_turbidity=args.use_turbidity,
            merge_boxes=args.merge_boxes
        )
        
        # Setup output directories
        self.out_root = Path(args.outdir)
        self.out_root.mkdir(parents=True, exist_ok=True)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary with pipeline results and paths
        """
        LOGGER.info("Starting vial detection pipeline...")
        
        # Stage A: Detect vials and create crops
        crops_dir, crop_records = self.run_vial_detection()
        
        # Stage B: Run liquid detection on crops
        liquid_out_dir = self.run_liquid_detection(crops_dir)
        
        # Stage C: Process results and create manifest
        manifest_path = self.process_results(crop_records, liquid_out_dir)
        
        # Generate summary report
        summary = self.generate_summary(manifest_path)
        
        LOGGER.info("Pipeline complete!")
        
        return {
            'crops_dir': str(crops_dir),
            'liquid_out_dir': str(liquid_out_dir),
            'manifest_path': str(manifest_path),
            'summary': summary
        }
    
    def run_vial_detection(self) -> Tuple[Path, List[Dict[str, Any]]]:
        """
        Stage A: Detect vials in images and create crops.
        
        Returns:
            Tuple of (crops_directory, crop_records)
        """
        LOGGER.info(f"[Stage A] Detecting vials in {self.args.source}")
        
        # Setup directories
        crops_dir = increment_path(self.out_root / "crops/exp", mkdir=True)
        
        # Load vial detection model
        vial_model = DetectMultiBackend(
            self.args.vial_weights, 
            device=self.device, 
            dnn=False, 
            data=None, 
            fp16=self.args.half
        )
        
        stride, names, pt = vial_model.stride, vial_model.names, vial_model.pt
        imgsz = self.args.imgsz * 2 if len(self.args.imgsz) == 1 else self.args.imgsz
        imgsz = check_img_size(imgsz, s=stride)
        
        # Process images
        dataset = LoadImages(self.args.source, img_size=imgsz, stride=stride, auto=pt)
        crop_records = []
        total_crops = 0
        
        for path, im, im0s, vid_cap, s in dataset:
            # Prepare image tensor
            im_tensor = torch.from_numpy(im).to(self.device)
            im_tensor = im_tensor.half() if vial_model.fp16 else im_tensor.float()
            im_tensor /= 255.0
            if im_tensor.ndim == 3:
                im_tensor = im_tensor[None]
            
            # Run inference
            pred = vial_model(im_tensor, augment=False, visualize=False)
            pred = non_max_suppression(
                pred, 
                self.args.vial_conf, 
                self.args.vial_iou, 
                max_det=1000
            )
            
            # Process detections
            im0 = im0s.copy()
            H, W = im0.shape[:2]
            
            for det in pred:
                if not len(det):
                    continue
                
                # Scale boxes to original image size
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                
                # Sort by confidence and keep top-k
                det = det[det[:, 4].argsort(descending=True)]
                if self.args.topk > 0:
                    det = det[:self.args.topk]
                
                # Extract crops
                for j, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Expand bounding box with padding
                    x1e, y1e, x2e, y2e = expand_and_clamp(
                        x1, y1, x2, y2, W, H, self.args.pad
                    )
                    
                    # Extract and resize crop
                    crop = im0[y1e:y2e, x1e:x2e].copy()
                    crop_resized, scale = resize_keep_height(crop, self.args.crop_h)
                    
                    # Save crop
                    stem = Path(path).stem
                    out_name = f"{stem}_vial{j:02d}.png"
                    out_path = crops_dir / out_name
                    cv2.imwrite(str(out_path), crop_resized)
                    
                    # Record crop information
                    record = {
                        "source_image": str(path),
                        "crop_image": str(out_path),
                        "vial_index": int(j),
                        "det_conf": float(conf),
                        "bbox_xyxy_src": [x1, y1, x2, y2],
                        "bbox_xyxy_expanded_src": [x1e, y1e, x2e, y2e],
                        "resize": {"target_h": int(self.args.crop_h), "scale": float(scale)},
                        "src_size": [int(W), int(H)],
                        "crop_size": [int(crop_resized.shape[1]), int(crop_resized.shape[0])]
                    }
                    crop_records.append(record)
                    total_crops += 1
        
        LOGGER.info(f"[Stage A] Saved {total_crops} crops to {crops_dir}")
        return crops_dir, crop_records
    
    def run_liquid_detection(self, crops_dir: Path) -> Path:
        """
        Stage B: Run liquid detection on crop images.
        
        Args:
            crops_dir: Directory containing crop images
            
        Returns:
            Path to liquid detection output directory
        """
        LOGGER.info(f"[Stage B] Running liquid detection on {crops_dir}")
        
        exp_name = crops_dir.name
        out_dir = self.out_root / "liquid_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command based on task type
        if self.args.liquid_task == "detect":
            script = "detect.py"
            cmd = [
                "python3", str(DEFAULT_PATHS['yolov5_root'] / script),
                "--weights", self.args.liquid_weights,
                "--source", str(crops_dir),
                "--imgsz", str(self.args.liquid_imgsz),
                "--conf", str(self.args.liquid_conf),
                "--iou", str(self.args.liquid_iou),
                "--save-txt", "--save-conf",
                "--project", str(out_dir),
                "--name", exp_name,
                "--line-thickness", "1",
                "--exist-ok",
            ]
        elif self.args.liquid_task == "segment":
            script = "segment/predict.py"
            cmd = [
                "python3", str(DEFAULT_PATHS['yolov5_root'] / script),
                "--weights", self.args.liquid_weights,
                "--source", str(crops_dir),
                "--imgsz", str(self.args.liquid_imgsz),
                "--conf", str(self.args.liquid_conf),
                "--iou", str(self.args.liquid_iou),
                "--project", str(out_dir),
                "--name", exp_name,
                "--exist-ok",
            ]
        else:
            raise ValueError(f"Unknown liquid_task: {self.args.liquid_task}")
        
        # Run the command
        LOGGER.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            LOGGER.error(f"Liquid detection failed: {result.stderr}")
            raise RuntimeError("Liquid detection failed")
        
        return out_dir / exp_name
    
    def process_results(self, crop_records: List[Dict[str, Any]], 
                       liquid_out_dir: Path) -> Path:
        """
        Stage C: Process detection results and create manifest.
        
        Args:
            crop_records: List of crop record dictionaries
            liquid_out_dir: Directory with liquid detection results
            
        Returns:
            Path to manifest file
        """
        LOGGER.info("[Stage C] Processing results and creating manifest")
        
        # Setup directories
        manifest_dir = liquid_out_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        turbidity_dir = liquid_out_dir / "turbidity"
        turbidity_dir.mkdir(parents=True, exist_ok=True)
        
        viz_dir = liquid_out_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = liquid_out_dir / "labels"
        
        # Process each crop
        exp_name = liquid_out_dir.name
        manifest_path = manifest_dir / f"manifest_{exp_name}.jsonl"
        
        state_counts = {
            "stable": 0,
            "gelled": 0,
            "phase_separated": 0,
            "only_air": 0,
            "unknown": 0
        }
        
        with open(manifest_path, "w") as fout:
            for record in crop_records:
                crop_path = Path(record["crop_image"])
                label_path = labels_dir / f"{crop_path.stem}.txt"
                
                # Check alternative locations for label file
                if not label_path.exists():
                    alt_paths = list(liquid_out_dir.parent.glob(f"exp*/labels/{crop_path.stem}.txt"))
                    if alt_paths:
                        label_path = alt_paths[-1]
                
                # Classify vial state
                state_info = self.classifier.classify(crop_path, label_path)
                
                # Add turbidity analysis if image exists
                img = cv2.imread(str(crop_path))
                if img is not None and self.args.save_turbidity:
                    profile = compute_turbidity_profile(img)
                    
                    # Save turbidity plot
                    plot_path = save_enhanced_turbidity_plot(
                        crop_path,
                        profile.normalized_profile,
                        getattr(profile, "excluded_regions", None),
                        turbidity_dir
                    )
                    state_info["turbidity_plot"] = str(plot_path)
                    
                    # Add turbidity statistics
                    state_info["turbidity_stats"] = {
                        "mean": float(np.mean(profile.normalized_profile)),
                        "std": float(np.std(profile.normalized_profile)),
                        "max_gradient": float(np.max(np.abs(np.gradient(profile.normalized_profile)))) 
                                       if len(profile.normalized_profile) > 1 else 0.0
                    }
                
                # Create visualization if requested
                # if self.args.save_viz and label_path.exists():
                #     viz_path = create_detection_visualization(
                #         crop_path,
                #         label_path,
                #         viz_dir / f"{crop_path.stem}_viz.jpg"
                #     )
                #     state_info["visualization"] = str(viz_path)
                
                # Combine all information
                record.update(state_info)
                fout.write(json.dumps(record) + "\n")
                
                # Update counts
                state = state_info.get("vial_state", "unknown")
                state_counts[state] = state_counts.get(state, 0) + 1
        
        # Print summary
        self._print_summary(state_counts, len(crop_records))
        
        return manifest_path
    
    def generate_summary(self, manifest_path: Path) -> Dict[str, Any]:
        """Generate summary statistics from manifest."""
        summary = {
            'total_vials': 0,
            'state_distribution': {},
            'turbidity_stats': {
                'mean': [],
                'std': [],
                'max_gradient': []
            }
        }
        
        with open(manifest_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                summary['total_vials'] += 1
                
                state = record.get('vial_state', 'unknown')
                summary['state_distribution'][state] = \
                    summary['state_distribution'].get(state, 0) + 1
                
                if 'turbidity_stats' in record:
                    for key in ['mean', 'std', 'max_gradient']:
                        if key in record['turbidity_stats']:
                            summary['turbidity_stats'][key].append(
                                record['turbidity_stats'][key]
                            )
        
        # Calculate averages
        stats_out = {}
        for key, values in summary['turbidity_stats'].items():
            if values:
                stats_out[f'{key}_avg'] = np.mean(values)
                stats_out[f'{key}_std'] = np.std(values)

        summary['turbidity_stats'] = stats_out

        return summary
    
    def _print_summary(self, state_counts: Dict[str, int], total: int):
        """Print processing summary."""
        print("\n" + "="*50)
        print("Processing Summary")
        print("="*50)
        print(f"Total vials processed: {total}")
        print("\nState Distribution:")
        for state, count in sorted(state_counts.items()):
            if count > 0:
                percentage = (count / total) * 100
                print(f"  {state:15s}: {count:4d} ({percentage:5.1f}%)")
        print("="*50 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vial detection and liquid analysis pipeline"
    )
    
    # Input/Output
    parser.add_argument("--source", type=str, required=True,
                       help="Input image/directory/video")
    parser.add_argument("--outdir", type=str, 
                       default=str(DEFAULT_PATHS['output_root']),
                       help="Output directory")
    
    # Stage A: Vial detection
    parser.add_argument("--vial-weights", type=str, required=True,
                       help="Path to vial detection model")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640],
                       help="Inference size (height, width)")
    parser.add_argument("--vial-conf", type=float, default=0.65,
                       help="Vial detection confidence threshold")
    parser.add_argument("--vial-iou", type=float, default=0.45,
                       help="Vial detection IoU threshold")
    parser.add_argument("--pad", type=float, default=0.12,
                       help="Padding fraction for vial crops")
    parser.add_argument("--crop-h", type=int, default=640,
                       help="Target height for vial crops")
    parser.add_argument("--topk", type=int, default=-1,
                       help="Keep top-K vials per image (-1 for all)")
    
    # Stage B: Liquid detection
    parser.add_argument("--liquid-task", choices=["detect", "segment"],
                       default="detect", help="Liquid detection task type")
    parser.add_argument("--liquid-weights", type=str, required=True,
                       help="Path to liquid detection model")
    parser.add_argument("--liquid-imgsz", type=int, default=640,
                       help="Liquid detection image size")
    parser.add_argument("--liquid-conf", type=float, default=0.25,
                       help="Liquid detection confidence threshold")
    parser.add_argument("--liquid-iou", type=float, default=0.50,
                       help="Liquid detection IoU threshold")
    
    # Analysis options
    parser.add_argument("--use-turbidity", action="store_true",
                       help="Use turbidity analysis for phase separation")
    parser.add_argument("--merge-boxes", action="store_true",
                       help="Merge overlapping detection boxes")
    parser.add_argument("--save-turbidity", action="store_true",
                       help="Save turbidity plots")
    parser.add_argument("--save-viz", action="store_true",
                       help="Save detection visualizations")
    
    # Hardware
    parser.add_argument("--device", type=str, default="",
                       help="CUDA device (0, 0,1,2,3 or cpu)")
    parser.add_argument("--half", action="store_true",
                       help="Use FP16 half-precision inference")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run pipeline
    pipeline = VialDetectionPipeline(args)
    results = pipeline.run()
    
    # Print final results
    print("\n" + "="*50)
    print("Summary Complete!")
    print("="*50)
    print(f"Crops directory:  {results['crops_dir']}")
    print(f"Liquid results:   {results['liquid_out_dir']}")
    print(f"Manifest file:    {results['manifest_path']}")
    print(f"Total vials:      {results['summary']['total_vials']}")
    print("="*50)


if __name__ == "__main__":
    main()