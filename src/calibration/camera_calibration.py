"""
camera_calibration.py
=====================
Performs camera calibration and radial distortion estimation
from a single checkerboard (planar grid) image.

Author: Navyashree N
Date: October 26, 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calibrate_camera_from_grid(image_path, pattern_size=(9, 6), square_size=1.0):
    """
    Estimate camera intrinsics and radial distortion parameters.

    Parameters
    ----------
    image_path : str
        Path to checkerboard or tiled grid image (.png, .jpg, etc.)
    pattern_size : tuple(int, int)
        Number of inner corners (cols, rows) → (9,6) for 10x7 grid.
    square_size : float
        Real-world size of each square (arbitrary units; consistent scale only).

    Returns
    -------
    dict : calibration results including camera matrix, distortion coeffs, and reprojection error.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        raise ValueError("⚠️ Checkerboard not detected! Check pattern_size or image clarity.")

    # Refine corner locations to subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Prepare 3D world coordinates (z=0 plane)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners_refined], gray.shape[::-1], None, None
    )

    # Compute reprojection error
    mean_error = 0
    for i in range(len(objp)):
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
        error = cv2.norm(corners_refined, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objp)

    # Undistort preview
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.title("📸 Distorted Input"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.subplot(1, 2, 2); plt.title("✨ Undistorted Output"); plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.savefig("results/calibration_result.png", bbox_inches='tight')
    plt.show()

    return {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "mean_reprojection_error": mean_error
    }

if __name__ == "__main__":
    results = calibrate_camera_from_grid("data/grid_image.png", pattern_size=(9, 6))
    print("\n🎯 Camera Intrinsics (K):\n", results["camera_matrix"])
    print("\n🔍 Distortion Coefficients [k1, k2, p1, p2, k3]:\n", results["dist_coeffs"].ravel())
    print("\n📏 Mean Reprojection Error: {:.4f} pixels".format(results["mean_reprojection_error"]))
