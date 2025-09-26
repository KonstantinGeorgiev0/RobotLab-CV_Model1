"""
Liquid detection module.
Handles Stage B: Running liquid detection on vial crops.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from yolov5.utils.general import LOGGER
from config import DEFAULT_PATHS


class LiquidDetector:
    """Detector for liquid content in vial crops."""
    
    def __init__(self,
                 weights_path: str,
                 task: str = 'detect',
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.50):
        """
        Initialize liquid detector.
        
        Args:
            weights_path: Path to liquid detection model
            task: Task type ('detect' or 'segment')
            img_size: Inference image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.weights_path = weights_path
        self.task = task
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Validate task
        if task not in ['detect', 'segment']:
            raise ValueError(f"Invalid task: {task}. Must be 'detect' or 'segment'")
        
        # Validate model exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        LOGGER.info(f"Liquid detector initialized: task={task}, model={weights_path}")
    
    def run_detection(self,
                     source_dir: Path,
                     output_dir: Path,
                     save_txt: bool = True,
                     save_conf: bool = True,
                     save_img: bool = True) -> Path:
        """
        Run liquid detection on directory of crops.
        
        Args:
            source_dir: Directory containing crop images
            output_dir: Directory to save results
            save_txt: Save detection labels
            save_conf: Save confidence values in labels
            save_img: Save annotated images
            
        Returns:
            Path to detection results directory
        """
        # Prepare output directory
        exp_name = source_dir.name
        results_dir = output_dir / exp_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = self._build_command(
            source_dir, 
            output_dir,
            exp_name,
            save_txt,
            save_conf,
            save_img
        )
        
        # Run detection
        LOGGER.info(f"Running liquid detection: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                LOGGER.error(f"Detection failed: {result.stderr}")
                raise RuntimeError(f"Liquid detection failed: {result.stderr}")
            
            LOGGER.info("Liquid detection complete")
            
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Detection process failed: {e}")
            raise
        
        return results_dir
    
    def _build_command(self,
                      source_dir: Path,
                      output_dir: Path,
                      exp_name: str,
                      save_txt: bool,
                      save_conf: bool,
                      save_img: bool) -> List[str]:
        """Build command for YOLOv5 detection script."""
        
        # Base script path
        if self.task == 'detect':
            script = DEFAULT_PATHS['yolov5_root'] / 'detect.py'
        else:  # segment
            script = DEFAULT_PATHS['yolov5_root'] / 'segment' / 'predict.py'
        
        # Build command
        cmd = [
            'python3', str(script),
            '--weights', str(self.weights_path),
            '--source', str(source_dir),
            '--imgsz', str(self.img_size),
            '--conf', str(self.conf_threshold),
            '--iou', str(self.iou_threshold),
            '--project', str(output_dir),
            '--name', exp_name,
            '--exist-ok',
            '--line-thickness', '1'
        ]
        
        # Add optional flags
        if save_txt:
            cmd.append('--save-txt')
        if save_conf and save_txt:
            cmd.append('--save-conf')
        if not save_img:
            cmd.append('--nosave')
        
        return cmd
    
    def parse_detection_results(self, 
                               labels_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse detection results from label files.
        
        Args:
            labels_dir: Directory containing YOLO format label files
            
        Returns:
            Dictionary mapping image names to detection lists
        """
        results = {}
        
        if not labels_dir.exists():
            LOGGER.warning(f"Labels directory not found: {labels_dir}")
            return results
        
        # Process each label file
        for label_file in labels_dir.glob('*.txt'):
            image_name = label_file.stem
            detections = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        det = {
                            'class_id': int(parts[0]),
                            'cx': float(parts[1]),
                            'cy': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4]),
                            'confidence': float(parts[5]) if len(parts) > 5 else 1.0
                        }
                        detections.append(det)
            
            results[image_name] = detections
        
        return results
    
    def get_detection_stats(self, 
                           labels_dir: Path) -> Dict[str, Any]:
        """
        Get statistics from detection results.
        
        Args:
            labels_dir: Directory containing label files
            
        Returns:
            Dictionary with detection statistics
        """
        all_detections = self.parse_detection_results(labels_dir)
        
        # Calculate statistics
        total_images = len(all_detections)
        total_detections = sum(len(dets) for dets in all_detections.values())
        
        # Count by class
        class_counts = {}
        confidence_values = []
        
        for detections in all_detections.values():
            for det in detections:
                class_id = det['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                confidence_values.append(det['confidence'])
        
        # Calculate confidence statistics
        if confidence_values:
            conf_stats = {
                'mean': float(np.mean(confidence_values)),
                'std': float(np.std(confidence_values)),
                'min': float(np.min(confidence_values)),
                'max': float(np.max(confidence_values))
            }
        else:
            conf_stats = {}
        
        return {
            'total_images': total_images,
            'total_detections': total_detections,
            'detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'class_distribution': class_counts,
            'confidence_stats': conf_stats
        }
