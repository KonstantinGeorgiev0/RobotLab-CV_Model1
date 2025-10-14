import numpy as np
import cv2
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("image", help="input vial crop")
ap.add_argument("--outdir", default="edge_sobel_results")
args = ap.parse_args()

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
img = cv2.imread(args.image, cv2.IMREAD_COLOR)
stem = Path(args.image).stem

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Original", image)
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

# cv2.imshow("Original Image | Laplacian", np.hstack([image, lap]))
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# cv2.imshow("Sobel X | Sobel Y | Sobel Combined", np.hstack([sobelX, sobelY, sobelCombined]))
# cv2.waitKey(0)
cv2.imwrite(str(outdir / f"result_{stem}_edge_image_sobelX.png"), sobelX)
cv2.imwrite(str(outdir / f"result_{stem}_edge_image_sobelY.png"), sobelY)
cv2.imwrite(str(outdir / f"result_{stem}_edge_image_combined.png"), sobelCombined)