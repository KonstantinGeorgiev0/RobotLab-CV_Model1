#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, cv2


def keep_near_horizontal(edges_u8, theta_deg=15):
    """Mask edges to only those with near-horizontal gradient orientation."""
    g = edges_u8 > 0
    if not g.any():
        return edges_u8
    # compute gradients on original grayscale neighborhood
    # we approximate by re-edge-detecting gradients on a blurred version of edges
    e = cv2.GaussianBlur(edges_u8, (5,5), 0)
    gx = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.degrees(np.arctan2(gy, gx))
    mask = (np.abs(ang) <= theta_deg)  # near horizontal
    out = np.zeros_like(edges_u8)
    out[mask] = edges_u8[mask]
    return out

def row_projection_peaks(bin_u8, min_sep_frac=0.04, prominence=0.15):
    """Find 1..N horizontal bands via row-wise edge counts."""
    H, W = bin_u8.shape
    proj = (bin_u8 > 0).sum(axis=1).astype(np.float32) / max(1, W)
    # smooth
    k = max(5, H//100)|1
    proj_s = cv2.GaussianBlur(proj.reshape(-1,1), (1,k), 0).ravel()
    # normalize
    p = (proj_s - proj_s.min()) / (proj_s.ptp() + 1e-6)
    # simple peak picking
    peaks = []
    i = 1
    while i < H-1:
        if p[i] > p[i-1] and p[i] > p[i+1] and p[i] >= prominence:
            peaks.append(i)
            # enforce min separation
            i += max(1, int(H*min_sep_frac))
        else:
            i += 1
    return peaks, p

def trace_curve_in_band(E, y_center, band_half=8, left_margin=0, right_margin=0):
    """Column-wise topmost trace but only inside [y_center±band_half]."""
    H, W = E.shape
    y_low  = max(0, int(y_center - band_half))
    y_high = min(H, int(y_center + band_half + 1))
    rows = np.arange(y_low, y_high)
    y = np.full(W, np.nan, np.float32)
    for x in range(left_margin, W-right_margin):
        col = E[y_low:y_high, x]
        ys = rows[col > 0]
        if ys.size:
            y[x] = ys.min()
    # interpolate & smooth
    x = np.arange(W, dtype=np.float32)
    m = ~np.isnan(y)
    if m.sum() < 2:
        return None, None, None
    y = np.interp(x, x[m], y[m])
    # median + box
    def medf(a, k=9):
        k |= 1; p = k//2
        pad = np.pad(a, (p,p), mode="edge")
        out = np.empty_like(a)
        for i in range(a.size): out[i] = np.median(pad[i:i+k])
        return out
    y = medf(y, 9)
    y = np.convolve(y, np.ones(11)/11, mode="same")
    # metrics
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = sol[0]*x + sol[1]
    wiggle = float(np.sqrt(np.mean((y - yhat)**2)) / max(1.0, H))
    dy = np.gradient(y); ddy = np.gradient(dy)
    curvature = float(np.mean(np.abs(ddy)) / max(1.0, H))
    return y, wiggle, curvature

def trace_multi_curves(img_bgr,
                       liquid_band=(0.30, 0.70),
                       wall_margin_frac=0.03,
                       canny=(60,140),
                       theta_horiz_deg=15,
                       band_half=10,
                       min_peak_sep_frac=0.04,
                       peak_prominence=0.15):
    """
    Returns list of dicts [{ 'y': array, 'wiggle':..., 'curv':..., 'y0': int }, ...]
    One entry per detected horizontal band within the liquid band.
    """
    H, W = img_bgr.shape[:2]
    # edges
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 60, 60)
    E = cv2.Canny(g, canny[0], canny[1], L2gradient=True)
    # mask search band and walls
    top = int(liquid_band[0]*H)
    bot = int(liquid_band[1]*H)
    lm  = int(wall_margin_frac*W)
    rm  = int(wall_margin_frac*W)
    mask = np.zeros_like(E, np.uint8); mask[top:bot, lm:W-rm] = 1
    E = (E & mask*255).astype(np.uint8)
    # keep near-horizontal only
    E = keep_near_horizontal(E, theta_deg=theta_horiz_deg)
    # row projection peaks
    peaks, _ = row_projection_peaks(E[top:bot, :], min_sep_frac=min_peak_sep_frac,
                                    prominence=peak_prominence)
    results = []
    for p in peaks:
        y_center = top + p
        y, wgl, curv = trace_curve_in_band(E, y_center, band_half=band_half, left_margin=lm, right_margin=rm)
        if y is not None:
            results.append({'y': y, 'wiggle': wgl, 'curv': curv, 'y0': int(y_center)})
    return results, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input vial crop (BGR or gray)")
    ap.add_argument("--outdir", default="trace_multi_curved_results")
    ap.add_argument("--canny", nargs=2, type=int, default=[60, 140],
                    help="Canny thresholds low high")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Could not read: {args.image}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)          # edge-preserving denoise
    edges = cv2.Canny(gray, args.canny[0], args.canny[1], L2gradient=True)

    curves, E_horiz = trace_multi_curves(img,
                                         liquid_band=(0.25, 0.85),
                                         wall_margin_frac=0.04,
                                         canny=(50, 130),
                                         theta_horiz_deg=12,
                                         band_half=8,
                                         min_peak_sep_frac=0.05,
                                         peak_prominence=0.12)

    overlay = img.copy()
    for c in curves:
        x = np.arange(overlay.shape[1])
        pts = np.vstack([x, c['y']]).T.astype(np.int32)
        for i in range(1, pts.shape[0]):
            cv2.line(overlay, tuple(pts[i - 1]), tuple(pts[i]), (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(overlay, f"wiggle={c['wiggle']:.3f} curv={c['curv']:.3f}",
        #             (10, max(20, c['y0'] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # cv2.imwrite("edge_detection/multi_curve_overlay.png", overlay)
    # cv2.imwrite("edge_detection/horiz_edge_mask.png", E_horiz)

    stem = Path(args.image).stem
    cv2.imwrite(str(outdir / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(outdir / f"{stem}_horiz_edge_mask.png"), E_horiz)
    cv2.imwrite(str(outdir / f"{stem}_curve_overlay.png"), overlay)
    print(f"Saved to {outdir}")

if __name__ == "__main__":
    main()
