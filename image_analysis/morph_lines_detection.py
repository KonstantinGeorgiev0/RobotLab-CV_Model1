#!/usr/bin/env python3
"""
morph_lines_detection.py
Headless version of OpenCV morphology line extraction.
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2 as cv

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_step(outdir: Path, prefix: str, name: str, img):
    p = outdir / f"{prefix}_{name}.png"
    cv.imwrite(str(p), img)
    return p

def to_gray(src):
    return cv.cvtColor(src, cv.COLOR_BGR2GRAY) if (src.ndim == 3) else src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--outdir", default="morph_lines_detection_results", help="Output folder")
    ap.add_argument("--prefix", default=None, help="Filename prefix (defaults to input stem)")
    ap.add_argument("--horiz-div", type=int, default=15,
                    help="Kernel divisor for horizontal size: width // HORIZ_DIV")
    ap.add_argument("--vert-div", type=int, default=30,
                    help="Kernel divisor for vertical size: height // VERT_DIV")
    ap.add_argument("--adaptive-block", type=int, default=15, help="Adaptive threshold block size")
    ap.add_argument("--adaptive-C", type=int, default=-2, help="Adaptive threshold C")
    args = ap.parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        print(f"Error: input not found: {in_path}")
        return 1

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    prefix = args.prefix if args.prefix else in_path.stem

    # --- Load & basic prep ---
    src = cv.imread(str(in_path), cv.IMREAD_COLOR)
    if src is None:
        print(f"Error opening image: {in_path}")
        return 1
    # save_step(outdir, prefix, "src", src)

    gray = to_gray(src)
    save_step(outdir, prefix, "gray", gray)

    # --- Binary (adaptive on inverted) ---
    inv = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(
        inv, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
        args.adaptive_block, args.adaptive_C
    )
    save_step(outdir, prefix, "binary", bw)

    # --- Init copies ---
    horizontal = np.copy(bw)
    vertical   = np.copy(bw)

    # --- Horizontal morphology ---
    cols = horizontal.shape[1]
    horizontal_size = max(3, cols // args.horiz_div)
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    save_step(outdir, prefix, "horizontal", horizontal)

    # --- Vertical morphology ---
    rows = vertical.shape[0]
    vertical_size = max(3, rows // args.vert_div)
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    save_step(outdir, prefix, "vertical", vertical)

    # --- Post "smoothing" block from the sample (kept for parity) ---
    vertical_bit = cv.bitwise_not(vertical)
    # save_step(outdir, prefix, "vertical_bit", vertical_bit)

    edges = cv.adaptiveThreshold(
        vertical_bit, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2
    )
    # save_step(outdir, prefix, "edges", edges)

    kernel = np.ones((2, 2), np.uint8)
    edges_dil = cv.dilate(edges, kernel)
    # save_step(outdir, prefix, "dilate", edges_dil)

    smooth = cv.blur(vertical_bit, (2, 2))
    # Copy smoothed pixels where edges_dil != 0
    vertical_smoothed = vertical_bit.copy()
    rr, cc = np.where(edges_dil != 0)
    vertical_smoothed[rr, cc] = smooth[rr, cc]
    # save_step(outdir, prefix, "smooth_final", vertical_smoothed)

    print(f"Saved outputs under: {outdir.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
