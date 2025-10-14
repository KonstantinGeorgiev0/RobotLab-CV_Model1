import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------- CONFIG ----------------------
IMG_PATH = '../data/bulk_test_cropped/Day_1_20w%_Proglyde_DMM_vial00.png'
CANNY_LOW, CANNY_HIGH = 50, 150
SOBEL_K, LAPL_K = 3, 3
DFT_LOW, DFT_HIGH = 6, 120

def ensure_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def add_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0,0), (len(text)*11+14, 26), (0,0,0), -1)
    cv2.putText(out, text, (8,19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out

def denoise_bilateral(img, d=7, sigma_color=50, sigma_space=50):
    if img.ndim == 3:
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    else:
        g3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.bilateralFilter(g3, d, sigma_color, sigma_space)

def flatfield_correct(img, kernel=51):
    gray = ensure_gray(img)
    bg = cv2.medianBlur(gray, kernel)
    bg = np.clip(bg, 1, None)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return to_bgr(norm)

def dft_filter(img, pass_type="bandpass", low=5, high=60, notch_list=None):
    g = ensure_gray(img)
    f = np.fft.fftshift(np.fft.fft2(g))
    H, W = g.shape
    Y, X = np.ogrid[:H, :W]
    cy, cx = H//2, W//2
    R = np.sqrt((Y-cy)**2 + (X-cx)**2)

    mask = np.ones_like(g, dtype=np.float32)
    if pass_type == "lowpass":
        mask = (R <= high).astype(np.float32)
    elif pass_type == "highpass":
        mask = (R >= low).astype(np.float32)
    elif pass_type == "bandpass":
        mask = ((R >= low) & (R <= high)).astype(np.float32)
    elif pass_type == "notch" and notch_list:
        mask = np.ones_like(g, dtype=np.float32)
        for (nx, ny, r) in notch_list:
            rr = np.hypot(X-(cx+nx), Y-(cy+ny))
            mask *= (rr > r).astype(np.float32)
            rr2 = np.hypot(X-(cx-nx), Y-(cy-ny))
            mask *= (rr2 > r).astype(np.float32)

    f_filt = f * mask
    out = np.fft.ifft2(np.fft.ifftshift(f_filt))
    out = np.abs(out)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return to_bgr(out)

def compute_edges(img, method="canny", **kw):
    g = ensure_gray(img)
    if method == "canny":
        low = kw.get("low", 50); high = kw.get("high", 150)
        return cv2.Canny(g, low, high, L2gradient=True)
    elif method == "sobel":
        k = kw.get("ksize", 3)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=k)
        # gx = np.uint8(np.absolute(gx))
        # gy = np.uint8(np.absolute(gy))
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        sobelCombined = cv2.bitwise_or(gx, gy)

        return cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    elif method == "laplacian":
        k = kw.get("ksize", 3)
        L = cv2.Laplacian(g, cv2.CV_32F, ksize=k)
        L = cv2.convertScaleAbs(L)
        return cv2.threshold(L, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        raise ValueError("Unknown edge method")

def make_montage(tiles, cols=3, pad=6, pad_color=(40,40,40)):
    Hs = [t.shape[0] for t in tiles]
    target_H = min(360, int(np.median(Hs)))
    norm_tiles = []
    max_W = 0
    for t in tiles:
        h, w = t.shape[:2]
        scale = target_H / float(h)
        tw = int(w*scale)
        resized = cv2.resize(t, (tw, target_H), interpolation=cv2.INTER_AREA)
        norm_tiles.append(resized)
        max_W = max(max_W, tw)
    padded = []
    for t in norm_tiles:
        h, w = t.shape[:2]
        if w < max_W:
            pad_w = max_W - w
            t = cv2.copyMakeBorder(t, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_color)
        padded.append(t)
    rows = int(np.ceil(len(padded) / cols))
    grid_rows = []
    idx = 0
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            if idx < len(padded):
                row_imgs.append(padded[idx])
            else:
                row_imgs.append(np.full((target_H, max_W, 3), pad_color, dtype=np.uint8))
            idx += 1
        row = cv2.hconcat(row_imgs)
        grid_rows.append(row)
    grid = cv2.vconcat(grid_rows)
    grid = cv2.copyMakeBorder(grid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=pad_color)
    return grid

def generate_demo_image(W=640, H=480):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Background gradient
    for y in range(H):
        val = int(50 + 50 * (y/H))
        img[y,:, :] = (val, val, val)
    # Air region (top)
    air_h = int(0.25 * H)
    img[:air_h, :, :] = 220
    # Liquid region with turbidity gradient
    for y in range(air_h, H-30):
        alpha = (y - air_h) / (H - air_h - 30)
        val = int(200 - 110*alpha + 15*np.sin(2*np.pi*alpha*6))
        img[y, :, :] = (val, val, val)
    # Bottom speckle to simulate sediment
    rng = np.random.default_rng(42)
    for _ in range(3500):
        y = int(rng.integers(low=H-60, high=H-10))
        x = int(rng.integers(low=20, high=W-20))
        col = int(rng.integers(80, 120))
        cv2.circle(img, (x, y), 1, (col, col, col), -1)
    # Faint horizontal interface
    y_int = air_h + int(0.48*(H-air_h-50))
    cv2.line(img, (40, y_int), (W-40, y_int), (180,180,180), 1, cv2.LINE_AA)
    # Mild Gaussian noise
    noise = rng.normal(0, 5, size=img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

# Load or demo
if IMG_PATH and Path(IMG_PATH).exists():
    src = cv2.imread(IMG_PATH)
    if src is None:
        src = generate_demo_image()
else:
    src = generate_demo_image()

# Pipeline
flat = flatfield_correct(src, kernel=51)
deno = denoise_bilateral(flat, d=7, sigma_color=60, sigma_space=60)
dft  = dft_filter(deno, pass_type="bandpass", low=DFT_LOW, high=DFT_HIGH)

edges_canny = compute_edges(dft, method="canny", low=CANNY_LOW, high=CANNY_HIGH)
edges_sobel = compute_edges(dft, method="sobel", ksize=SOBEL_K)
edges_lap   = compute_edges(dft, method="laplacian", ksize=LAPL_K)

tiles = [
    add_label(to_bgr(src), "Original"),
    add_label(to_bgr(ensure_gray(src)), "Original (Gray)"),
    add_label(to_bgr(ensure_gray(flat)), "Flatfield Corrected"),
    add_label(to_bgr(ensure_gray(deno)), "Bilateral Denoised"),
    add_label(to_bgr(ensure_gray(dft)),  "DFT Bandpass"),
    add_label(to_bgr(edges_canny), "Canny"),
    add_label(to_bgr(edges_sobel), "Sobel (Otsu)"),
    add_label(to_bgr(edges_lap),   "Laplacian (Otsu)")
]

montage = make_montage(tiles, cols=3, pad=8)

out_dir = Path("results_all_methods_edges")
out_dir.mkdir(parents=True, exist_ok=True)

paths = {}
paths["montage"] = str(out_dir / "edge_viz_montage.png")
cv2.imwrite(paths["montage"], montage)

pairs = [
    ("01_original.png", to_bgr(src)),
    ("02_original_gray.png", to_bgr(ensure_gray(src))),
    ("03_flatfield.png", to_bgr(ensure_gray(flat))),
    ("04_bilateral.png", to_bgr(ensure_gray(deno))),
    ("05_dft_bandpass.png", to_bgr(ensure_gray(dft))),
    ("06_edges_canny.png", to_bgr(edges_canny)),
    ("07_edges_sobel.png", to_bgr(edges_sobel)),
    ("08_edges_laplacian.png", to_bgr(edges_lap)),
]
for fname, img in pairs:
    p = out_dir / fname
    cv2.imwrite(str(p), img)
    paths[fname] = str(p)

# Show montage (single figure)
plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
plt.show()
