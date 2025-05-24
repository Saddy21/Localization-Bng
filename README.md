# FOD Localization and Mapping


<p align="center">
  <a href="https://arxiv.org/abs/1812.03506"><img src="doc/hloc.png" width="60%"/></a>
  <br /><em>Hierachical Localization uses both image retrieval and feature matching</em>
</p>

##

## Installation

`hloc` requires Python >=3.7 and PyTorch >=1.1. Installing the package locally pulls the other dependencies:

```bash
git clone --recursive https://github.com/Saddy21/Localization-Bng.git
cd Localization-Bng/
python -m pip install -e .
```

All dependencies are listed in `requirements.txt`.  This repository includes external local features as git submodules – don't forget to pull submodules with `git submodule update --init --recursive`.

For Docker image:
```bash
docker build -t hloc:latest .
docker run -it --rm -p 8888:8888 hloc:latest  # for GPU support, add `--runtime=nvidia`
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

## General pipeline

# FOD 3D Mapping and Image Localization

This repository contains scripts for creating 3D maps from images and localizing new query images within those maps using [Hierarchical Localization (HLoc)](https://github.com/cvg/Hierarchical-Localization).

## Overview

The workflow consists of two main steps:
1. **Training**: Build a 3D map from a set of reference images
2. **Localization**: Estimate the 6DOF camera pose of new query images within the pre-built map

## Requirements

### Installation

```bash
# Clone HLoc repository
git clone https://github.com/cvg/Hierarchical-Localization.git
cd Hierarchical-Localization

# Install dependencies
pip install -e .

# Install additional requirements
pip install matplotlib pillow
```

### System Requirements
- Python 3.7+
- CUDA-capable GPU (recommended for faster feature extraction)
- At least 8GB RAM
- Sufficient disk space for feature storage

## Dataset Structure

Organize your data as follows:

```
project/
├── datasets/
│   └── your_scene/
│       ├── mapping/          # Reference images for building the map
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       └── query/            # Query images to localize
│           ├── query1.jpg
│           ├── query2.jpg
│           └── ...
├── outputs/                  # Generated outputs
└── scripts/
    ├── training_script.py
    └── localization_script.py
```

## 1. Training (Building the 3D Map)

### Basic Usage

```bash
python training_script.py /path/to/mapping/images --output /path/to/outputs
```

### Full Command Options

```bash
python training_script.py [mapping_images_dir] [options]

Options:
  --output OUTPUT_DIR     Output directory for results (default: ./outputs)
  --features FEATURES     Feature extractor: disk, superpoint, sift (default: disk)
  --matcher MATCHER       Feature matcher: lightglue, superglue, nnmatcher (default: lightglue)
  --max-keypoints N       Maximum keypoints per image (default: 5000)
  --resize-max N          Maximum image dimension for processing (default: 1600)
  --no-visualize          Skip visualization steps
  --no-save               Don't save intermediate files
```

### Example Commands

```bash
# Basic training with default settings
python training_script.py ../datasets/sacre_coeur/mapping

# Custom output directory and features
python training_script.py ../datasets/sacre_coeur/mapping \
    --output ../outputs/sacre_coeur \
    --features superpoint \
    --matcher superglue

# High-quality settings for challenging scenes
python training_script.py ../datasets/my_scene/mapping \
    --max-keypoints 8000 \
    --resize-max 2400

# Fast processing (skip visualizations)
python training_script.py ../datasets/my_scene/mapping \
    --no-visualize
```

## 2. Localization (Query Image Pose Estimation)

### Basic Usage

```bash
python localization_script.py /path/to/query/image.jpg /path/to/model_info.pkl
```

### Full Command Options

```bash
python localization_script.py [query_image] [model_info.pkl] [options]

Options:
  --output OUTPUT_DIR     Output directory for results
  --no-visualize          Skip visualization steps
  --no-save              Don't save result files
  --no-cleanup           Don't remove temporary files
  --min-inliers N        Minimum inliers for reliable pose (default: 12)
  --max-error FLOAT      Maximum RANSAC error (default: 12.0)
  --batch                Process all images in query directory
```

### Example Commands

```bash
# Localize single image
python localization_script.py ../datasets/sacre_coeur/query/night.jpg ../outputs/model_info.pkl

# Batch process all images in query folder
python localization_script.py ../datasets/sacre_coeur/query/ ../outputs/model_info.pkl --batch

# Custom output location
python localization_script.py query.jpg model_info.pkl --output ./results/my_localization

# Relaxed settings for challenging images
python localization_script.py difficult_query.jpg model_info.pkl \
    --min-inliers 8 \
    --max-error 15.0

# Fast processing without visualizations
python localization_script.py query.jpg model_info.pkl --no-visualize
```

### Localization Outputs

For each query image, the script generates:
- `localization_results.pkl` - Complete Python results object
- `localization_results.json` - JSON format for easy parsing
- `camera_pose.txt` - Human-readable pose information
- `query_image.png` - Query image visualization
- `correspondences_2d.png` - 2D feature correspondences
- `localization_3d.html` - Interactive 3D visualization

### Understanding Results

#### Success Criteria
- **Successful**: ≥12 inliers with good geometric consistency
- **Uncertain**: <12 inliers or poor geometric fit
- **Failed**: No pose could be estimated

#### Pose Information
- **Position**: 3D coordinates (x, y, z) in the map coordinate system
- **Orientation**: Quaternion (w, x, y, z) representing camera rotation
- **Inlier Ratio**: Percentage of feature matches that support the pose
- **Reprojection Error**: Average pixel error of 3D-2D correspondences

#### Example Output
```
✓ Localization succeeded:
  - Inlier correspondences: 3114/4621
  - Inlier ratio: 67.4%
  - Camera position (x, y, z): [-1.693, -0.627, -2.881]
  - Camera orientation (quaternion w, x, y, z): [-0.049270, 0.053042, -0.013726, 0.997282]
```

## Complete Workflow Example

```bash
# 1. Prepare your dataset
mkdir -p datasets/my_scene/{mapping,query}
# Copy your reference images to datasets/my_scene/mapping/
# Copy your query images to datasets/my_scene/query/

# 2. Build the 3D map
python training_script.py datasets/my_scene/mapping --output outputs/my_scene

# 3. Localize query images
python localization_script.py datasets/my_scene/query/ outputs/my_scene/model_info.pkl --batch

# 4. View results
ls outputs/my_scene/  # Training outputs
ls datasets/my_scene/query/*/  # Individual localization results
```

## Troubleshooting

### Common Issues

**Training fails with "Insufficient matches"**
- Ensure images have good visual overlap
- Try different feature extractors (--features superpoint)
- Check image quality and lighting consistency

**Localization fails with "No pose estimated"**
- Query image may be too different from training images
- Try relaxed parameters: `--min-inliers 6 --max-error 20.0`
- Ensure query image shows the same scene as training images

**Out of memory errors**
- Reduce `--max-keypoints` (try 2000-3000)
- Reduce `--resize-max` (try 1200-1400)
- Process fewer images at once

**Slow processing**
- Use `--no-visualize` for faster processing
- Ensure CUDA is available for GPU acceleration
- Reduce image resolution with `--resize-max`

### Performance Tips

- **GPU Usage**: HLoc automatically uses GPU if available
- **Memory Management**: Close visualization windows to free memory
- **Batch Processing**: More efficient than processing images individually
- **Feature Caching**: Features are cached and reused when possible

## Advanced Configuration

### Feature Extractors
- **DISK** (default): Good balance of speed and accuracy
- **SuperPoint**: Robust to lighting changes
- **SIFT**: Classical method, good for textured scenes

### Matchers
- **LightGlue** (default): Fast and accurate
- **SuperGlue**: High accuracy for challenging conditions
- **NN Matcher**: Simple nearest neighbor matching

### Quality vs Speed Trade-offs
- **High Quality**: More keypoints, larger images, SuperPoint+SuperGlue
- **High Speed**: Fewer keypoints, smaller images, DISK+LightGlue
- **Balanced**: Default settings work well for most scenes

## File Formats

### model_info.pkl Structure
```python
{
    'sfm_dir': 'path/to/reconstruction',
    'images_path': 'path/to/images',
    'features_path': 'path/to/features',
    'matches_path': 'path/to/matches',
    'feature_conf': {...},
    'matcher_conf': {...},
    'references_registered': [list of image names]
}
```

### Localization Results JSON
```json
{
    "success": true,
    "num_inliers": 3114,
    "total_matches": 4621,
    "inlier_ratio": 0.674,
    "camera_position": [-1.693, -0.627, -2.881],
    "camera_quaternion": [-0.049270, 0.053042, -0.013726, 0.997282],
    "camera_params": {
        "model": "SIMPLE_RADIAL",
        "width": 2816,
        "height": 2112,
        "params": [2735.54, 1408.0, 1056.0, -0.001234]
    }
}
```



- **COLMAP/pycolmap**: Check the [COLMAP documentation](https://colmap.github.io/)
