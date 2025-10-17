import sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, cv2
from image_analysis.line_hv_detection import detect_vial_wall_xcoords_by_components
from robotlab_utils.image_utils import extract_edges_for_curve_detection


def _rolling_median(y, k=9):
    k = max(3, k | 1)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=np.float32)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+k])
    return out

def _hampel_1d(y, k=7, n_sig=3.0):
    """Classic Hampel filter for isolated outliers."""
    k = max(3, k | 1); pad = k//2
    ypad = np.pad(y, (pad,pad), mode='edge')
    out = y.copy()
    for i in range(len(y)):
        win = ypad[i:i+k]
        med = np.median(win)
        mad = np.median(np.abs(win - med)) + 1e-6
        if abs(y[i] - med) > n_sig * 1.4826 * mad:
            out[i] = med
    return out

def _kill_short_runs(y, max_run=3, jump_px=8):
    """
    Remove tiny 'pikes': if a run deviates > jump_px from local median but lasts <= max_run,
    replace it by surrounding median.
    """
    med = _rolling_median(y, k=9)
    out = y.copy()
    i = 0
    N = len(y)
    while i < N:
        if abs(y[i] - med[i]) > jump_px:
            j = i
            while j+1 < N and abs(y[j+1] - med[j+1]) > jump_px:
                j += 1
            if (j - i + 1) <= max_run:
                out[i:j+1] = med[i:j+1]
            i = j + 1
        else:
            i += 1
    return out


def trace_surface_within_vial_using_edges(
    img_bgr: np.ndarray,
    y_band: tuple[int, int] | None = None,   # vertical limits
    x_band: tuple[int, int] | None = None,   # optional horizontal override
    sidewall_pad_px: int = 6,
    median_k: int = 9,
    max_step_px: int = 4
):
    H, W = img_bgr.shape[:2]

    # --- 1) Horizontal bounds from REAL walls ---
    xL, xR = detect_vial_wall_xcoords_by_components(
        img_bgr,
        left_strip_frac=0.14, right_strip_frac=0.14,
        min_vert_span_frac=0.40,
        y_band=(int(0.12*H), int(0.90*H)) if y_band is None else y_band
    )
    if x_band is not None:
        xL = max(0, min(xL, x_band[0]))
        xR = min(W-1, max(xR, x_band[1]))

    # tiny inward pad to avoid the glass highlight; keep small to see more width
    x0 = max(0, xL + max(0, sidewall_pad_px))
    x1 = min(W-1, xR - max(0, sidewall_pad_px))
    xs = np.arange(x0, x1+1, dtype=int)
    if xs.size < 5:
        return xs, np.array([], dtype=np.float32), (xL, xR)

    # --- 2) Vertical band (ignore cap/bottom) ---
    if y_band is None:
        y_min, y_max = int(0.25*H), int(0.70*H)
    else:
        y_min, y_max = max(0, y_band[0]), min(H-1, y_band[1])
        if y_max <= y_min:
            y_min, y_max = int(0.25*H), int(0.70*H)

    # --- 3) Edges tuned for the wavy surface (your function) ---
    edges = extract_edges_for_curve_detection(img_bgr)

    # --- 4) Column-wise pick inside [y_min, y_max] ---
    ys = []
    band_h = y_max - y_min + 1
    w = cv2.getGaussianKernel(band_h, band_h/6).reshape(-1)
    for x in xs:
        col = edges[y_min:y_max+1, x]
        if not np.any(col):
            ys.append((y_min + y_max)//2)
            continue

        nz = np.flatnonzero(col)
        y_top = y_min + int(nz[0])  # first hit from top (air->liquid)
        if len(nz) > 4:
            votes = (col > 0).astype(np.float32) * w
            yw = np.sum(votes * np.arange(y_min, y_max+1))
            sw = np.sum(votes) + 1e-6
            y_centroid = yw / sw
            y_pick = 0.6*y_top + 0.4*y_centroid
        else:
            y_pick = float(y_top)
        ys.append(y_pick)
    ys = np.array(ys, dtype=np.float32)

    # --- 5) De-spike chain: median -> Hampel -> short-run killer -> slope clip ---
    ys = _rolling_median(ys, k=median_k)
    ys = _hampel_1d(ys, k=7, n_sig=3.0)
    ys = _kill_short_runs(ys, max_run=3, jump_px=8)

    # slope clipping to enforce continuity
    for i in range(1, len(ys)):
        d = ys[i] - ys[i-1]
        if d > max_step_px:   ys[i] = ys[i-1] + max_step_px
        if d < -max_step_px:  ys[i] = ys[i-1] - max_step_px

    return xs, ys, (xL, xR)


def draw_surface(img_bgr, xs, ys, x_bounds, color=(0,255,0), thickness=2):
    out = img_bgr.copy()
    for i in range(1, len(xs)):
        cv2.line(out, (int(xs[i-1]), int(ys[i-1])), (int(xs[i]), int(ys[i])), color, thickness, cv2.LINE_AA)
    # visualize vial bounds (debug)
    cv2.line(out, (x_bounds[0], 0), (x_bounds[0], out.shape[0]-1), (0,200,255), 1, cv2.LINE_AA)
    cv2.line(out, (x_bounds[1], 0), (x_bounds[1], out.shape[0]-1), (0,200,255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-o", "--outdir", default="curve_traces_results", help="output directory for results")
    ap.add_argument("--ymin", type=float, default=0.20, help="search band start as frac of H")
    ap.add_argument("--ymax", type=float, default=0.80, help="search band end as frac of H")
    ap.add_argument("--xmin", type=float, default=0.05, help="search band start as frac of W")
    ap.add_argument("--xmax", type=float, default=0.95, help="search band end as frac of W")
    ap.add_argument("--pad", type=int, default=1, help="sidewall padding (px)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    H, W = img.shape[:2]
    y_band = (int(args.ymin * H), int(args.ymax * H))
    x_band = (int(args.xmin * W), int(args.xmax * W))

    xs, ys, bounds = trace_surface_within_vial_using_edges(
        img_bgr=img,
        y_band=y_band,
        x_band=x_band,
        sidewall_pad_px=6,
        median_k=9,
        max_step_px=4
    )

    overlay = draw_surface(img, xs, ys, bounds)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(outdir / f"{stem}_curve_overlay.png"), overlay)
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()