#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, cv2

from robotlab_utils.image_utils import extract_edges_for_curve_detection


def trace_curve_on_edges(edges, left_margin_frac=0.055, right_margin_frac=0.055,
                         top_frac=0.25, bottom_frac=0.85):
    H, W = edges.shape
    # connect fragments with a horizontal close
    k = max(3, W // 80)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    E = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kh, iterations=1)

    # mask margins & vertical search band
    lm = int(left_margin_frac * W)
    rm = int(right_margin_frac * W)
    top = int(top_frac * H)
    bot = int(bottom_frac * H)

    mask = np.zeros_like(E, np.uint8)
    mask[top:bot, lm:W-rm] = 1
    E[mask == 0] = 0

    # column-wise topmost edge
    rows = np.arange(H, dtype=np.int32)
    y = np.full(W, np.nan, np.float32)
    for x in range(lm, W-rm):
        ys = rows[E[:, x] > 0]
        if ys.size: y[x] = ys.min()

    # fill gaps + smooth (median then box)
    x = np.arange(W, dtype=np.float32)
    m = ~np.isnan(y)
    if m.sum() >= 2:
        y = np.interp(x, x[m], y[m])
    else:
        return None, None, None

    def medfilt(a, k=9):
        k |= 1; p = k//2
        pad = np.pad(a, (p,p), mode="edge")
        out = np.empty_like(a)
        for i in range(a.size): out[i] = np.median(pad[i:i+k])
        return out

    y = medfilt(y, 9)
    y = np.convolve(y, np.ones(11)/11, mode="same")

    # metrics
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = sol[0]*x + sol[1]
    wiggle = float(np.sqrt(np.mean((y - yhat)**2)) / max(1.0, H))
    dy = np.gradient(y); ddy = np.gradient(dy)
    curvature = float(np.mean(np.abs(ddy)) / max(1.0, H))

    return y, wiggle, curvature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input vial crop (BGR or gray)")
    ap.add_argument("--outdir", default="trace_curve_results")
    ap.add_argument("--canny", nargs=2, type=int, default=[15, 70],
                    help="Canny thresholds low high")
    ap.add_argument("--denoise", choices=['bilateral', 'epf', 'nlmeans', 'gaussian', 'none'],
                    default='bilateral', help="Denoising method")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Could not read: {args.image}")

    # Use centralized edge extraction function with configurable denoising
    edges = extract_edges_for_curve_detection(
        img,
        canny_low=args.canny[0],
        canny_high=args.canny[1],
        denoise_method=args.denoise,
        morphology_close=True,
        morphology_dilate=True
    )

    y, wiggle, curvature = trace_curve_on_edges(edges)
    overlay = img.copy()
    if y is not None:
        x = np.arange(overlay.shape[1])
        pts = np.vstack([x, y]).T.astype(np.int32)
        for i in range(1, pts.shape[0]):
            cv2.line(overlay, tuple(pts[i-1]), tuple(pts[i]), (0,255,0), 2, cv2.LINE_AA)
        txt = f"curve  wiggle={wiggle:.3f}  curv={curvature:.4f}"
    else:
        txt = "curve not found"

    cv2.putText(overlay, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    stem = Path(args.image).stem
    cv2.imwrite(str(outdir / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(outdir / f"{stem}_curve_overlay.png"), overlay)
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
