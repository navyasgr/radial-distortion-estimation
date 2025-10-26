"# data folder" 
# Data Folder

This folder contains all input images used for radial distortion calibration.

## Structure
- `raw/` : Original images of planar grids or checkerboards.
- `sample/` : Sample images provided for testing the pipeline.
- `augmented/` : Optional: Augmented images (rotated, scaled, or occluded) for robustness tests.

## Usage
- Place all images here before running calibration scripts.
- Scripts in `src/` will automatically read images from this folder.



