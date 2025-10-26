"""
detect_grid_points.py
=====================

Module for robust detection of planar grid corners in a single image.

Handles:
- Checkerboard / planar grid detection
- Sub-pixel refinement
- Occlusion and noise resilience
- Returns 2D coordinates suitable for calibration

Author: Navyashree N
Date: 2025-10-26
"""

from typing import List, Tuple
import cv2
import numpy as np


class GridDetector:
    """
    Detects corners of a planar rectangular grid in a single image.

    Attributes:
        image (np.ndarray): Input grayscale or color image.
        grid_size (Tuple[int, int]): Expected number of inner corners (rows, cols).
    """

    def __init__(self, image: np.ndarray, grid_size: Tuple[int, int] = None):
        self.image = image
        self.grid_size = grid_size
        self.corners = None

    def detect_corners(self) -> np.ndarray:
        """
        Detect grid corners using OpenCV's findChessboardCorners with fallback.

        Returns:
            np.ndarray: Array of (x, y) coordinates of detected corners.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) \
            if len(self.image.shape) == 3 else self.image

        if self.grid_size:
            ret, corners = cv2.findChessboardCorners(gray, self.grid_size, None)
            if ret:
                # Sub-pixel refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                self.corners = corners.reshape(-1, 2)
                return self.corners

        # Fallback: Harris corners + clustering
        self.corners = self._harris_corners(gray)
        return self.corners

    def _harris_corners(self, gray: np.ndarray) -> np.ndarray:
        """
        Fallback Harris corner detection for partial or occluded grids.

        Args:
            gray (np.ndarray): Grayscale image.

        Returns:
            np.ndarray: Detected corners.
        """
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst_thresh = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst_thresh = np.uint8(dst_thresh)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)
        return centroids


# Example Usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "../data/sample/grid1.jpg"
    img = cv2.imread(image_path)
    detector = GridDetector(img, grid_size=(9, 6))
    corners = detector.detect_corners()

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(corners[:, 0], corners[:, 1], color='red', s=10)
    plt.title("Detected Grid Corners")
    plt.show()
