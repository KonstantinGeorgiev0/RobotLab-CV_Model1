"""
Vial detection module.
Handles Stage A: Detecting vials in images and creating crops.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    non_max_suppression, scale_boxes, check_img_size, 
    LOGGER, increment_path
)
from yolov5.utils.torch_utils import select_device

from robotlab_utils.bbox_utils import expand_and_clamp
from robotlab_utils.image_utils import resize_keep_height


class VialDetector:
    """Detector for vials in images."""
    
    def __init__(self, 
                 weights_path: str,
                 device: str = '',
                 img_size: int = 640,
                 conf_threshold: float = 0.65,
                 iou_threshold: float = 0.45,
                 half_precision: bool = False):
        """
        Initialize vial detector.
        
        Args:
            weights_path: Path to model weights
            device: CUDA device string
            img_size: Inference image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            half_precision: Use FP16 inference
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half_precision = half_precision
        
        # Initialize device
        self.device = select_device(device)
        
        # Load model
        self.model = DetectMultiBackend(
            weights_path,
            device=self.device,
            dnn=False,
            data=None,
            fp16=half_precision
        )
        
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        
        # Check and adjust image size
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        self.img_size = check_img_size(img_size, s=self.stride)
        
        LOGGER.info(f"Vial detector initialized with model: {weights_path}")
    
    def detect_and_crop(self,
                       source: str,
                       output_dir: Path,
                       pad_fraction: float = 0.12,
                       crop_height: int = 640,
                       top_k: int = -1) -> Tuple[Path, List[Dict[str, Any]]]:
        """
        Detect vials in images and create crops.
        
        Args:
            source: Path to input images/directory/video
            output_dir: Directory to save crops
            pad_fraction: Padding to add around vial bbox
            crop_height: Target height for crops
            top_k: Keep top-k detections per image (-1 for all)
            
        Returns:
            Tuple of (crops_directory, list of crop records)
        """
        # Setup output directory
        crops_dir = increment_path(output_dir / "crops/exp", mkdir=True)
        LOGGER.info(f"Saving crops to: {crops_dir}")
        
        # Load dataset
        dataset = LoadImages(
            source, 
            img_size=self.img_size, 
            stride=self.stride, 
            auto=self.pt
        )
        
        crop_records = []
        total_detections = 0
        
        # Process each image
        for path, im, im0s, vid_cap, s in dataset:
            # Preprocess
            im_tensor = self._preprocess_image(im)
            
            # Run inference
            detections = self._run_inference(im_tensor)
            
            # Process detections
            im0 = im0s.copy()
            records = self._process_detections(
                detections,
                im_tensor.shape,
                im0,
                path,
                crops_dir,
                pad_fraction,
                crop_height,
                top_k
            )
            
            crop_records.extend(records)
            total_detections += len(records)
            
            LOGGER.info(f"Processed {path}: {len(records)} vials detected")
        
        LOGGER.info(f"Total vials detected: {total_detections}")
        return crops_dir, crop_records
    
    def _preprocess_image(self, im: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        im_tensor = torch.from_numpy(im).to(self.device)
        im_tensor = im_tensor.half() if self.model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        
        if im_tensor.ndim == 3:
            im_tensor = im_tensor[None]  # Add batch dimension
        
        return im_tensor
    
    def _run_inference(self, im_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Run model inference."""
        pred = self.model(im_tensor, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            classes=None,
            agnostic=False,
            max_det=1000
        )
        return pred
    
    def _process_detections(self,
                           detections: List[torch.Tensor],
                           tensor_shape: tuple,
                           original_image: np.ndarray,
                           image_path: str,
                           crops_dir: Path,
                           pad_fraction: float,
                           crop_height: int,
                           top_k: int) -> List[Dict[str, Any]]:
        """Process detections and create crops."""
        records = []
        H, W = original_image.shape[:2]
        
        for det in detections:
            if not len(det):
                continue
            
            # Scale boxes to original image
            det[:, :4] = scale_boxes(tensor_shape[2:], det[:, :4], original_image.shape).round()
            
            # Sort by confidence and keep top-k
            det = det[det[:, 4].argsort(descending=True)]
            if top_k > 0:
                det = det[:top_k]
            
            # Process each detection
            for j, (*xyxy, conf, cls) in enumerate(det):
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Expand bounding box
                x1e, y1e, x2e, y2e = expand_and_clamp(
                    x1, y1, x2, y2, W, H, pad_fraction
                )
                
                # Extract crop
                crop = original_image[y1e:y2e, x1e:x2e].copy()
                
                # Resize crop
                crop_resized, scale = resize_keep_height(crop, crop_height)
                
                # Save crop
                stem = Path(image_path).stem
                crop_name = f"{stem}_vial{j:02d}.png"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop_resized)
                
                # Create record
                record = {
                    "source_image": str(image_path),
                    "crop_image": str(crop_path),
                    "vial_index": int(j),
                    "det_conf": float(conf),
                    "class_id": int(cls),
                    "bbox_xyxy_src": [x1, y1, x2, y2],
                    "bbox_xyxy_expanded_src": [x1e, y1e, x2e, y2e],
                    "resize": {
                        "target_h": int(crop_height),
                        "scale": float(scale)
                    },
                    "src_size": [int(W), int(H)],
                    "crop_size": [int(crop_resized.shape[1]), int(crop_resized.shape[0])]
                }
                records.append(record)
        
        return records
    
    def warmup(self):
        """Warmup model for better performance."""
        LOGGER.info("Warming up model...")
        dummy = torch.zeros(1, 3, *self.img_size).to(self.device)
        dummy = dummy.half() if self.model.fp16 else dummy.float()
        
        for _ in range(3):
            _ = self.model(dummy, augment=False, visualize=False)
        
        LOGGER.info("Model warmup complete")