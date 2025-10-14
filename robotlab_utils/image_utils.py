"""
Image processing utility functions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_keep_height(img: np.ndarray, target_h: int) -> Tuple[np.ndarray, float]:
    """
    Resize image to target height while maintaining aspect ratio.
    
    Args:
        img: Input image
        target_h: Target height
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = img.shape[:2]
    
    if h == target_h:
        return img, 1.0
    
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def extract_region(img: np.ndarray, bbox: list, 
                   min_size: int = 10) -> Optional[np.ndarray]:
    """
    Extract region from image based on bounding box.
    
    Args:
        img: Source image
        bbox: Bounding box [x1, y1, x2, y2]
        min_size: Minimum dimension size to consider valid
        
    Returns:
        Cropped region or None if too small
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    
    # Extract region
    region = img[y1:y2, x1:x2].copy()
    
    # Check minimum size
    if region.shape[0] < min_size or region.shape[1] < min_size:
        return None
    
    return region


def calculate_brightness_stats(img: np.ndarray) -> dict:
    """
    Calculate brightness statistics for an image.
    
    Args:
        img: Input image (can be BGR or grayscale)
        
    Returns:
        Dictionary with brightness statistics
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    return {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'median': float(np.median(gray))
    }


def apply_adaptive_threshold(img: np.ndarray, 
                           block_size: int = 11,
                           C: float = 2) -> np.ndarray:
    """
    Apply adaptive threshold to image.
    
    Args:
        img: Input image
        block_size: Size of pixel neighborhood for threshold calculation
        C: Constant subtracted from weighted mean
        
    Returns:
        Binary thresholded image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    
    return thresh


def enhance_contrast_CLAHE(img: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        img: Input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space if color image
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Apply CLAHE directly to grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(img)
    
    return enhanced


def sobel_edge_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lap = cv2.Laplacian(img, cv2.CV_64F)
    # lap = np.uint8(np.absolute(lap))
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    return sobelCombined