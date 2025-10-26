"""
distortion_model.py
===================

Module implementing the Division Distortion Model for radial lens distortion.

Features:
- Forward distortion and iterative undistortion
- Handles wide-angle and fisheye lenses
- Vectorized operations for high performance
- Suitable for integration into calibration pipelines

Author: Navyashree N
Date: 2025-10-26
"""

from typing import Tuple
import numpy as np


class DivisionDistortionModel:
    """
    Division model for radial distortion:
        x_d = x_u / (1 + k1*r^2 + k2*r^4)
        y_d = y_u / (1 + k1*r^2 + k2*r^4)
    """

    @staticmethod
    def distort(points: np.ndarray, k1: float, k2: float, cx: float, cy: float) -> np.ndarray:
        """
        Forward distortion of undistorted points.

        Args:
            points (np.ndarray): Nx2 array of (x, y) coordinates.
            k1, k2 (float): Radial distortion coefficients.
            cx, cy (float): Principal point coordinates.

        Returns:
            np.ndarray: Nx2 array of distorted points.
        """
        x = points[:, 0] - cx
        y = points[:, 1] - cy
        r2 = x**2 + y**2
        denom = 1 + k1*r2 + k2*r2**2
        xd = x / denom + cx
        yd = y / denom + cy
        return np.vstack((xd, yd)).T

    @staticmethod
    def undistort(points: np.ndarray, k1: float, k2: float, cx: float, cy: float,
                  iterations: int = 10, tol: float = 1e-6) -> np.ndarray:
        """
        Iteratively undistort points using Newton-Raphson method.

        Args:
            points (np.ndarray): Nx2 array of distorted points.
            k1, k2 (float): Radial distortion coefficients.
            cx, cy (float): Principal point coordinates.
            iterations (int): Max iterations for convergence.
            tol (float): Tolerance for convergence.

        Returns:
            np.ndarray: Nx2 array of undistorted points.
        """
        xd = points[:, 0] - cx
        yd = points[:, 1] - cy
        xu = np.copy(xd)
        yu = np.copy(yd)

        for i in range(iterations):
            r2 = xu**2 + yu**2
            denom = 1 + k1*r2 + k2*r2**2
            xu_new = xd * denom
            yu_new = yd * denom

            # Check convergence
            if np.max(np.abs(xu_new - xu)) < tol and np.max(np.abs(yu_new - yu)) < tol:
                break

            xu, yu = xu_new, yu_new

        xu += cx
        yu += cy
        return np.vstack((xu, yu)).T


# Example Usage
if __name__ == "__main__":
    points = np.array([[100, 200], [300, 400], [500, 600]], dtype=float)
    k1, k2 = -0.2, 0.05
    cx, cy = 320, 240

    distorted = DivisionDistortionModel.distort(points, k1, k2, cx, cy)
    undistorted = DivisionDistortionModel.undistort(distorted, k1, k2, cx, cy)

    print("Original Points:\n", points)
    print("Distorted Points:\n", distorted)
    print("Recovered Points:\n", undistorted)
