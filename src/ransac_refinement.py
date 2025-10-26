"""
RANSAC + Nonlinear Optimization for Robust Radial Distortion Estimation
-----------------------------------------------------------------------
This module refines the estimated distortion parameters by eliminating
outliers and minimizing the reprojection error.

Author: Navyashree N
Institution: IITM
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

def project_points(points_3d, rvec, tvec, K, dist_coeffs):
    """Projects 3D points to 2D using given camera parameters."""
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    return projected.reshape(-1, 2)

def reprojection_error(params, points_3d, points_2d, K):
    """Computes the reprojection error given camera params."""
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    dist_coeffs = params[6:]
    projected = project_points(points_3d, rvec, tvec, K, dist_coeffs)
    return (projected - points_2d).ravel()

def ransac_refine(points_3d, points_2d, K, dist_coeffs, iterations=100, threshold=1.0):
    """Performs RANSAC-based robust fitting of distortion parameters."""
    n_points = len(points_2d)
    best_inliers = []
    best_params = None

    for i in range(iterations):
        sample_idx = np.random.choice(n_points, size=max(6, n_points // 2), replace=False)
        p3d_sample = points_3d[sample_idx]
        p2d_sample = points_2d[sample_idx]

        init_params = np.hstack([
            np.zeros(3),  # rvec
            np.zeros(3),  # tvec
            dist_coeffs   # initial distortion
        ])

        # Fit on sampled data
        result = least_squares(reprojection_error, init_params,
                               args=(p3d_sample, p2d_sample, K),
                               verbose=0)

        # Compute residuals for all points
        residuals = reprojection_error(result.x, points_3d, points_2d, K)
        residuals = np.sqrt(residuals[::2]**2 + residuals[1::2]**2)

        inliers = np.where(residuals < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = result.x

    print(f"\n✅ Best model found with {len(best_inliers)} / {n_points} inliers")

    # Final refinement with all inliers
    refined = least_squares(reprojection_error, best_params,
                            args=(points_3d[best_inliers], points_2d[best_inliers], K),
                            verbose=1)

    return refined.x, best_inliers
