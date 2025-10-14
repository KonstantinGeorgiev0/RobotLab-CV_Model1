#!/usr/bin/env python3
# lines_probabilistic_hough_detection.py (headless)
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import cv2


OUTDIR = Path("curved_line_detection_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

def waviness_from_segments(segments, w_norm):
    """
    'wiggle' metric: fit a straight line y=ax+b over
    all segment midpoints, compute RMS deviation normalized by width.
    """
    if not segments:
        return 0.0
    mids = []
    for (x1,y1,x2,y2) in segments:
        mids.append(((x1+x2)/2.0, (y1+y2)/2.0))
    mids = np.asarray(mids, dtype=np.float32)
    X = np.hstack([mids[:,0:1], np.ones((mids.shape[0],1), np.float32)])  # [x, 1]
    y = mids[:,1]
    # least squares y = a*x + b
    sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = (sol[0]*mids[:,0] + sol[1])
    rms = float(np.sqrt(np.mean((y - y_hat)**2)))
    wiggle = rms / max(1.0, w_norm)
    return wiggle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--top-margin", type=float, default=0.1,
                        help="fraction of image height to exclude from top (default: 0.1)")
    parser.add_argument("--bottom-margin", type=float, default=0.1,
                        help="fraction of image height to exclude from bottom (default: 0.1)")
    args = parser.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read: {img_path}")

    # exclude top and bottom regions of image based on parameters
    H, W = img.shape[:2]
    top_pixels = int(H * args.top_margin)
    bottom_pixels = int(H * args.bottom_margin)

    # Create mask for valid image region
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top_pixels:H - bottom_pixels, :] = 255

    # Apply mask to image
    img = cv2.bitwise_and(img, img, mask=mask)

    # 1) Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilateral filter
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    # Apply stronger Gaussian blur to merge nearby wave components
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Detect edges using Canny with lower thresholds to catch faint gel waves
    edges = cv2.Canny(enhanced, 15, 70)

    # Apply stronger morphological closing to connect wave segments better
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    # Dilate more to make waves thicker and more connected
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel_dilate, iterations=2)

    # 2) Probabilistic Hough
    # Correct signature: HoughLinesP(image, rho, theta, threshold, minLineLength=..., maxLineGap=...)
    minLineLength = max(30, int(0.1 * W))
    maxLineGap = 5
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=minLineLength, maxLineGap=maxLineGap)

    segments = []
    if raw is not None:
        for s in raw:
            x1,y1,x2,y2 = map(int, s[0])
            segments.append((x1,y1,x2,y2))

    # 3) (Optional) keep only near-horizontal segments (good for interfaces)
    #    |angle| <= 10 degrees
    horiz = []
    for (x1,y1,x2,y2) in segments:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(ang) <= 10.0:
            horiz.append((x1,y1,x2,y2))

    # 4) Waviness metric (smaller ~ straighter line)
    wiggle = waviness_from_segments(horiz, W)

    # 5) Create overlay and save
    overlay = img.copy()
    for (x1,y1,x2,y2) in horiz:
        cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"horiz_lines={len(horiz)}  wiggle={wiggle:.3f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

    stem = Path(args.image).stem
    cv2.imwrite(str(OUTDIR / f"{stem}_curved_edges.png"), edges)
    cv2.imwrite(str(OUTDIR / f"{stem}_curved_overlay.png"), overlay)
    print(f"Saved: {OUTDIR/'curved_edges.png'}")
    print(f"Saved: {OUTDIR/'curved_overlay.png'}")
    print(f"Detected {len(horiz)} near-horizontal segments; wiggle={wiggle:.3f}")

if __name__ == "__main__":
    main()

