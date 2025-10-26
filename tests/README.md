"# tests folder" 
# Tests Folder

This folder contains unit tests and validation scripts for the calibration pipeline.

## Contents
- `test_detect_grid_points.py` : Tests for grid detection module.
- `test_distortion_model.py` : Tests for division distortion model.
- `test_ransac_outlier.py` : Tests for adaptive RANSAC module.
- `test_calibrator.py` : End-to-end pipeline validation.

## Purpose
- Ensures correctness, reproducibility, and robustness.
- Edge cases are included (partial occlusion, noise, extreme distortion).
- Run tests using `pytest` or any Python test runner.


