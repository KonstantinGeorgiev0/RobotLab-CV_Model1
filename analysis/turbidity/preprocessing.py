import sys
from pathlib import Path

import cv2
import numpy as np
from dataclasses import dataclass

# Import existing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from config import TURBIDITY_PARAMS

@dataclass
class PreprocessResult:
    original_image: np.ndarray
    original_height: int
    original_width: int
    analysis_image: np.ndarray
    height: int
    width: int
    gray: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    raw_profile: np.ndarray


def preprocess_img(image: np.ndarray) -> PreprocessResult:
    h_original, w_original = image.shape[:2]
    # resize to analysis dimensions
    analysis_img = cv2.resize(image,
                              (TURBIDITY_PARAMS['analysis_width'],
                               TURBIDITY_PARAMS['analysis_height']))

    height, width = analysis_img.shape[:2]
    # different color spaces
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)
    # row wise mean intensity
    raw_profile = np.mean(gray, axis=1)

    return PreprocessResult(
        original_image=image,
        original_height=h_original,
        original_width=w_original,
        analysis_image=analysis_img,
        height=height,
        width=width,
        gray=gray,
        hsv=hsv,
        lab=lab,
        raw_profile=raw_profile
    )