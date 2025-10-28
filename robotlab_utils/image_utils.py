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
        target_h: Target height in pixels
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = img.shape[:2]
    
    if h == target_h:
        return img, 1.0
    
    scale = target_h / h
    target_w = int(w * scale)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
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
                           tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.

    Args:
        img: Input image (BGR or grayscale)
        clip_limit: CLAHE clip limit
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image in original color space
    """
    if img.ndim == 2:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)
    else:
        # Color image - work in LAB space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def edge_preserving_smooth(img: np.ndarray,
                           method: str = "bilateral",
                           bilateral_params: Optional[dict] = None,
                           epf_params: Optional[dict] = None,
                           nlmeans_params: Optional[dict] = None,
                           gaussian_params: Optional[dict] = None) -> np.ndarray:
    """
    Apply edge-preserving smoothing to reduce noise while maintaining edges.

    Args:
        img: Input image (BGR or grayscale)
        method: Smoothing method to use:
            - 'bilateral': Bilateral filter (fast, good for most cases)
            - 'epf': Edge Preserving Filter (slower, preserves edges better)
            - 'nlmeans': Non-Local Means Denoising (slowest, best quality)
            - 'gaussian': Standard Gaussian blur (fastest, not edge-preserving)
    """
    if img is None:
        raise ValueError("Input image is None")

    if method == "bilateral":
        params = bilateral_params or {'d': 7, 'sigmaColor': 75, 'sigmaSpace': 75}
        return cv2.bilateralFilter(img, **params)

    elif method == "epf":
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            params = epf_params or {'flags': 1, 'sigma_s': 60, 'sigma_r': 0.4}
            result = cv2.edgePreservingFilter(img_bgr, **params)
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            params = epf_params or {'flags': 1, 'sigma_s': 60, 'sigma_r': 0.4}
            return cv2.edgePreservingFilter(img, **params)

    elif method == "nlmeans":
        params = nlmeans_params or {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                img, None,
                params['h'], params['h'],
                params['templateWindowSize'],
                params['searchWindowSize']
            )
        else:
            return cv2.fastNlMeansDenoising(
                img, None,
                params['h'],
                params['templateWindowSize'],
                params['searchWindowSize']
            )

    elif method == "gaussian":
        params = gaussian_params or {'ksize': (7, 7), 'sigmaX': 0}
        return cv2.GaussianBlur(img, **params)

    else:
        raise ValueError(f"Unknown smoothing method: {method}. "
                         f"Choose from: 'bilateral', 'epf', 'nlmeans', 'gaussian'")


def preprocess_for_edge_detection(img: np.ndarray,
                                  denoise_method: str = 'bilateral',
                                  enhance_contrast: bool = True,
                                  bilateral_params: Optional[dict] = None,
                                  gaussian_params: Optional[dict] = None,
                                  epf_params: Optional[dict] = None,
                                  nlmeans_params: Optional[dict] = None,
                                  clahe_params: Optional[dict] = None) -> np.ndarray:
    """
    Preprocess image for edge detection with configurable pipeline.

    Args:
        img: Input image (BGR or grayscale)
        denoise_method: Denoising method ('bilateral', 'epf', 'nlmeans', 'gaussian', or 'none')
        enhance_contrast: Whether to apply CLAHE contrast enhancement
        bilateral_params: Parameters for bilateral filter
        gaussian_params: Parameters for Gaussian blur
        epf_params: Parameters for edge preserving filter
        nlmeans_params: Parameters for non-local means denoising
        clahe_params: Parameters for CLAHE

    Returns:
        Preprocessed grayscale image ready for edge detection
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply denoising
    if denoise_method != 'none':
        params_map = {
            'bilateral': bilateral_params,
            'gaussian': gaussian_params,
            'epf': epf_params,
            'nlmeans': nlmeans_params
        }

        kwargs = {f"{denoise_method}_params": params_map.get(denoise_method)}
        gray = edge_preserving_smooth(gray, method=denoise_method, **{k: v for k, v in kwargs.items() if v is not None})

    # Apply contrast enhancement if requested
    if enhance_contrast:
        params = clahe_params or {'clipLimit': 2.0, 'tileGridSize': (8, 8)}
        clahe = cv2.createCLAHE(clipLimit=params['clipLimit'],
                                tileGridSize=params['tileGridSize'])
        gray = clahe.apply(gray)

    return gray


def extract_edges_for_curve_detection(img: np.ndarray,
                                      canny_low: int = 15,
                                      canny_high: int = 70,
                                      denoise_method: str = 'bilateral',
                                      morphology_close: bool = True,
                                      morphology_dilate: bool = True) -> np.ndarray:
    """
    Extract edges optimized for detecting curved patterns (gel waves, phase boundaries).

    This function combines preprocessing and edge detection specifically tuned for
    detecting non-straight features like gel waves and phase separation boundaries
    in vial images.

    Args:
        img: Input image (BGR or grayscale)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        denoise_method: Denoising method ('bilateral', 'epf', 'nlmeans', 'gaussian', or 'none')
        morphology_close: Apply morphological closing to connect edge segments
        morphology_dilate: Apply dilation to thicken edges

    Returns:
        Binary edge map (uint8, values 0 or 255)
    """

    # Preprocess: denoise + enhance contrast
    gray = preprocess_for_edge_detection(
        img,
        denoise_method=denoise_method,
        enhance_contrast=True,
        gaussian_params={'ksize': (7, 7), 'sigmaX': 0},
        clahe_params={'clipLimit': 2.0, 'tileGridSize': (8, 8)}
    )

    # Detect edges with lower thresholds to catch faint gel waves
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Apply morphological operations to connect wave segments
    if morphology_close:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    if morphology_dilate:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.dilate(edges, kernel_dilate, iterations=2)

    return edges

# Guided curve helpers
def compute_search_region_from_guide(
        H: int,
        W: int,
        horizontal_bounds: tuple[float, float],
        search_offset_frac: float,
        guide_y_px: int
    ) -> tuple[int, int, int, int]:
    """
    Compute (x_min_px, x_max_px, y_min_search, y_max_search) from bounds and a guide y.
    """
    x_min_px = int(horizontal_bounds[0] * W)
    x_max_px = int(horizontal_bounds[1] * W)
    x_min_px = max(0, min(x_min_px, W - 1))
    x_max_px = max(0, min(x_max_px, W - 1))
    if x_max_px < x_min_px:
        x_min_px, x_max_px = x_max_px, x_min_px

    search_offset_px = int(search_offset_frac * H)
    y_min_search = max(0, guide_y_px - search_offset_px)
    y_max_search = min(H - 1, guide_y_px + search_offset_px)
    if y_max_search < y_min_search:
        y_min_search, y_max_search = y_max_search, y_min_search

    return x_min_px, x_max_px, y_min_search, y_max_search


def vertical_sobel_edge_magnitude(
        gray_roi: np.ndarray,
        gaussian_ksize: tuple[int, int] = (3, 3),
        sobel_ksize: int = 5,
        equalize_hist: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    blur -> (optional) equalize -> Sobel dy -> abs -> normalize to 0..255 -> Otsu.
    Returns (edges_uint8, otsu_binary_uint8).
    """
    if gray_roi.ndim != 2:
        raise ValueError("vertical_sobel_edge_magnitude expects a grayscale ROI")

    # Apply blur for noise reduction
    if gaussian_ksize is not None:
        gray_proc = cv2.GaussianBlur(gray_roi, gaussian_ksize, 0)
    else:
        gray_proc = gray_roi.copy()

    # Contrast for low-light vials
    if equalize_hist:
        gray_proc = cv2.equalizeHist(gray_proc)

    # Vertical gradient using Sobel operator
    sobel_y = cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag = np.abs(sobel_y)

    # Normalize safely to [0, 255]
    mmax = float(mag.max())
    if mmax > 0:
        edges = (mag * (255.0 / mmax)).astype(np.uint8)
    else:
        edges = np.zeros_like(gray_proc, dtype=np.uint8)

    # Otsu threshold for a debug/visual binary mask
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges, thresh


def build_curve_edges_from_guide(
        img_bgr: np.ndarray,
        guide_y_px: int,
        horizontal_bounds: tuple[float, float],
        search_offset_frac: float,
        gaussian_ksize: tuple[int, int] = (3, 3),
        sobel_ksize: int = 5,
        equalize_hist: bool = True
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
      1) computes the search region from the guide,
      2) crops the ROI,
      3) runs the vertical-Sobel edge magnitude pipeline.

    Returns:
        edges (uint8), otsu_binary (uint8), (x_min_px, x_max_px, y_min_search, y_max_search)
    """
    H, W = img_bgr.shape[:2]
    x_min_px, x_max_px, y_min_search, y_max_search = compute_search_region_from_guide(
        H, W, horizontal_bounds, search_offset_frac, guide_y_px
    )

    # Prepare grayscale ROI
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray_roi = gray_full[y_min_search:y_max_search + 1, x_min_px:x_max_px + 1]

    edges, thresh = vertical_sobel_edge_magnitude(
        gray_roi,
        gaussian_ksize=gaussian_ksize,
        sobel_ksize=sobel_ksize,
        equalize_hist=equalize_hist
    )

    return edges, thresh, (x_min_px, x_max_px, y_min_search, y_max_search)


# Guided curve helpers
def compute_search_region_from_guide(
        H: int,
        W: int,
        horizontal_bounds: tuple[float, float],
        search_offset_frac: float,
        guide_y_px: int
    ) -> tuple[int, int, int, int]:
    """
    Compute (x_min_px, x_max_px, y_min_search, y_max_search) from bounds and a guide y.
    """
    x_min_px = int(horizontal_bounds[0] * W)
    x_max_px = int(horizontal_bounds[1] * W)
    x_min_px = max(0, min(x_min_px, W - 1))
    x_max_px = max(0, min(x_max_px, W - 1))
    if x_max_px < x_min_px:
        x_min_px, x_max_px = x_max_px, x_min_px

    search_offset_px = int(search_offset_frac * H)
    y_min_search = max(0, guide_y_px - search_offset_px)
    y_max_search = min(H - 1, guide_y_px + search_offset_px)
    if y_max_search < y_min_search:
        y_min_search, y_max_search = y_max_search, y_min_search

    return x_min_px, x_max_px, y_min_search, y_max_search


def vertical_sobel_edge_magnitude(
        gray_roi: np.ndarray,
        gaussian_ksize: tuple[int, int] = (3, 3),
        sobel_ksize: int = 5,
        equalize_hist: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    blur -> (optional) equalize -> Sobel dy -> abs -> normalize to 0..255 -> Otsu.
    Returns (edges_uint8, otsu_binary_uint8).
    """
    if gray_roi.ndim != 2:
        raise ValueError("vertical_sobel_edge_magnitude expects a grayscale ROI")

    # Apply blur for noise reduction
    if gaussian_ksize is not None:
        gray_proc = cv2.GaussianBlur(gray_roi, gaussian_ksize, 0)
    else:
        gray_proc = gray_roi.copy()

    # Contrast for low-light vials
    if equalize_hist:
        gray_proc = cv2.equalizeHist(gray_proc)

    # Vertical gradient using Sobel operator
    sobel_y = cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag = np.abs(sobel_y)

    # Normalize safely to [0, 255]
    mmax = float(mag.max())
    if mmax > 0:
        edges = (mag * (255.0 / mmax)).astype(np.uint8)
    else:
        edges = np.zeros_like(gray_proc, dtype=np.uint8)

    # Otsu threshold for a debug/visual binary mask
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges, thresh


def build_curve_edges_from_guide(
        img_bgr: np.ndarray,
        guide_y_px: int,
        horizontal_bounds: tuple[float, float],
        search_offset_frac: float,
        gaussian_ksize: tuple[int, int] = (3, 3),
        sobel_ksize: int = 5,
        equalize_hist: bool = True
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
      1) computes the search region from the guide,
      2) crops the ROI,
      3) runs the vertical-Sobel edge magnitude pipeline.

    Returns:
        edges (uint8), otsu_binary (uint8), (x_min_px, x_max_px, y_min_search, y_max_search)
    """
    H, W = img_bgr.shape[:2]
    x_min_px, x_max_px, y_min_search, y_max_search = compute_search_region_from_guide(
        H, W, horizontal_bounds, search_offset_frac, guide_y_px
    )

    # Prepare grayscale ROI
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray_roi = gray_full[y_min_search:y_max_search + 1, x_min_px:x_max_px + 1]

    edges, thresh = vertical_sobel_edge_magnitude(
        gray_roi,
        gaussian_ksize=gaussian_ksize,
        sobel_ksize=sobel_ksize,
        equalize_hist=equalize_hist
    )

    return edges, thresh, (x_min_px, x_max_px, y_min_search, y_max_search)

