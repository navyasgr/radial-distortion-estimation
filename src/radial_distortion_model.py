"""
radial_distortion_model.py
-----------------------------------
Implements the Brown–Conrady radial distortion model for camera calibration.

Author: Navyashree N
Date: October 2025
Institute: IIT Madras - Computer Vision   Project

This module defines mathematical functions to model, apply, and
remove radial distortion using least-squares estimation.
"""

import numpy as np
from typing import Tuple

def apply_radial_distortion(points: np.ndarray, k1: float, k2: float, k3: float = 0.0) -> np.ndarray:
    """
    Apply radial distortion to a set of normalized image points.

    Parameters:
        points (np.ndarray): Nx2 array of undistorted (x, y) points
        k1 (float): First radial distortion coefficient
        k2 (float): Second radial distortion coefficient
        k3 (float): Third radial distortion coefficient (optional)

    Returns:
        np.ndarray: Nx2 array of distorted (x_d, y_d) points
    """
    x, y = points[:, 0], points[:, 1]
    r2 = x**2 + y**2
    radial_factor = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)

    x_d = x * radial_factor
    y_d = y * radial_factor

    return np.column_stack((x_d, y_d))


def remove_radial_distortion(points: np.ndarray, k1: float, k2: float, k3: float = 0.0, iterations: int = 5) -> np.ndarray:
    """
    Iteratively correct (undistort) image points based on estimated distortion coefficients.

    Parameters:
        points (np.ndarray): Nx2 array of distorted (x_d, y_d) points
        k1, k2, k3 (float): Radial distortion coefficients
        iterations (int): Number of refinement iterations

    Returns:
        np.ndarray: Nx2 array of undistorted (x, y) points
    """
    x_d, y_d = points[:, 0], points[:, 1]
    x_u, y_u = x_d.copy(), y_d.copy()

    for _ in range(iterations):
        r2 = x_u**2 + y_u**2
        radial_factor = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
        x_u = x_d / radial_factor
        y_u = y_d / radial_factor

    return np.column_stack((x_u, y_u))
