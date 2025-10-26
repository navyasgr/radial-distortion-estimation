"""
plot_results.py
====================
Comprehensive visualization for radial distortion calibration.

Features:
- Original vs undistorted image
- Residual error per corner
- Radial distortion heatmap
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------
# Set up project paths
# ------------------------
DATA_PATH = "data/grid_image.png"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------
# Load image
# ------------------------
img = cv2.imread(DATA_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------------
# Detect corners
# ------------------------
pattern_size = (9,6)
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
if not ret:
    raise RuntimeError("Checkerboard corners not found!")

# Draw detected corners on original
img_corners = img.copy()
cv2.drawChessboardCorners(img_corners, pattern_size, corners, ret)

# Save original with corners
original_path = os.path.join(RESULTS_DIR, "original_corners.png")
cv2.imwrite(original_path, img_corners)
print(f"✅ Original corners saved at {original_path}")

# ------------------------
# Camera matrix & distortion coefficients (from Step 7 results)
# ------------------------
camera_matrix = np.array([[2847.2, 0, 1952.3],
                          [0, 2851.8, 1468.7],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([-0.287435, 0.092156, 0, 0, 0], dtype=np.float64)

# ------------------------
# Undistort image
# ------------------------
undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
undistorted_path = os.path.join(RESULTS_DIR, "undistorted.png")
cv2.imwrite(undistorted_path, undistorted)
print(f"✅ Undistorted image saved at {undistorted_path}")

# ------------------------
# Residuals per corner
# ------------------------
projected = cv2.undistortPoints(corners, camera_matrix, dist_coeffs)
projected = projected.reshape(-1,2)
corners_np = corners.reshape(-1,2)
residuals = np.linalg.norm(corners_np - projected, axis=1)

plt.figure(figsize=(10,4))
plt.bar(range(len(residuals)), residuals, color='skyblue')
plt.title("Residual Error per Corner")
plt.xlabel("Corner Index")
plt.ylabel("Residual (pixels)")
residuals_path = os.path.join(RESULTS_DIR, "residuals.png")
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Residuals plot saved at {residuals_path}")

# ------------------------
# Radial distortion heatmap
# ------------------------
h, w = img.shape[:2]
X, Y = np.meshgrid(np.arange(w), np.arange(h))
x = (X - camera_matrix[0,2]) / camera_matrix[0,0]
y = (Y - camera_matrix[1,2]) / camera_matrix[1,1]
r2 = x**2 + y**2
k1, k2 = dist_coeffs[0], dist_coeffs[1]
radial_magnitude = np.abs(k1*r2 + k2*r2**2)

plt.figure(figsize=(8,6))
plt.imshow(radial_magnitude, cmap="hot")
plt.colorbar(label="Radial Distortion Magnitude")
plt.title("🔥 Radial Distortion Heatmap")
plt.axis("off")
heatmap_path = os.path.join(RESULTS_DIR, "distortion_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Distortion heatmap saved at {heatmap_path}")
