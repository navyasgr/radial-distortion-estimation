"""
grid_corner_detection.py
-----------------------------------
Detects the inner corners of a checkerboard or rectangular grid using OpenCV.

Author: Navyashree N
Date: October 2025
Institute: IIT Madras - Computer Vision Project

Description:
------------
This script reads a single image of a planar rectangular grid and detects
the inner corner points. These corner coordinates are later used to estimate
camera parameters and correct for radial distortion.

Dependencies:
-------------
- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def detect_grid_corners(image_path: str, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the grid corners from a checkerboard or tiled grid image.

    Parameters:
        image_path (str): Path to the grid image (e.g., 'data/grid_image.png')
        pattern_size (Tuple[int, int]): Number of inner corners per row and column (e.g., (7, 7))

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - corners: Nx2 array of detected corner coordinates
            - refined_corners: Nx2 array of subpixel refined corner coordinates
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print("⚠️  Checkerboard corners not detected. Try adjusting pattern size or lighting.")
        return None, None

    # Refine corner coordinates to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw and display
    vis_img = cv2.drawChessboardCorners(img.copy(), pattern_size, refined_corners, ret)

    plt.figure(figsize=(7, 7))
    plt.title("✅ Detected Grid Corners")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig("results/corner_detection_output.png", bbox_inches='tight')

    plt.show()

    return corners.squeeze(), refined_corners.squeeze()


if __name__ == "__main__":
    # Example usage (replace with your actual grid image path)
    corners, refined = detect_grid_corners("data/grid_image.png", (9, 6))

    if refined is not None:
        print("Detected", len(refined), "corners successfully.")
