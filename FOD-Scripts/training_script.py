#!/usr/bin/env python
# coding: utf-8

"""
3D Map Building Script for Visual Localization
This script builds a 3D map from reference images using Structure-from-Motion.
The resulting model can be used for localizing query images.
"""

import argparse
import os
import sys
from pathlib import Path
import pickle

# Import HLoc modules
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


def setup_paths(dataset_path, output_path):
    """Setup directory paths for the mapping process."""
    images = Path(dataset_path)
    outputs = Path(output_path)
    
    # Create output directory
    outputs.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    sfm_pairs = outputs / 'pairs-sfm.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    model_info = outputs / 'model_info.pkl'
    
    return {
        'images': images,
        'outputs': outputs,
        'sfm_pairs': sfm_pairs,
        'sfm_dir': sfm_dir,
        'features': features,
        'matches': matches,
        'model_info': model_info
    }


def get_reference_images(images_path, mapping_subdir='mapping'):
    """Get list of reference images for mapping."""
    mapping_dir = images_path / mapping_subdir
    if not mapping_dir.exists():
        # If no mapping subdirectory, use all images in the main directory
        mapping_dir = images_path
    
    references = [str(p.relative_to(images_path)) for p in mapping_dir.iterdir() 
                  if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
    
    if not references:
        raise ValueError(f"No images found in {mapping_dir}")
    
    return references


def build_3d_map(images_path, output_path, feature_type='disk', matcher_type='disk+lightglue', 
                 visualize=True, save_visualization=True):
    """
    Build a 3D map from reference images.
    
    Args:
        images_path: Path to directory containing reference images
        output_path: Path to save the 3D model and intermediate files
        feature_type: Type of features to extract ('disk', 'superpoint', 'sift', etc.)
        matcher_type: Type of matcher to use ('disk+lightglue', 'superglue', etc.)
        visualize: Whether to display visualizations
        save_visualization: Whether to save visualization plots
    
    Returns:
        model: The reconstructed 3D model
    """
    
    print("Setting up paths...")
    paths = setup_paths(images_path, output_path)
    
    print("Configuring feature extractor and matcher...")
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]
    
    print("Getting reference images...")
    references = get_reference_images(paths['images'])
    print(f"Found {len(references)} reference images:")
    for ref in references:
        print(f"  - {ref}")
    
    if visualize:
        print("Visualizing reference images...")
        try:
            plot_images([read_image(paths['images'] / r) for r in references], dpi=10)
            if save_visualization:
                import matplotlib.pyplot as plt
                plt.savefig(paths['outputs'] / 'reference_images.png', dpi=150, bbox_inches='tight')
                plt.show()
        except Exception as e:
            print(f"Warning: Could not visualize images: {e}")
    
    print("Extracting features from reference images...")
    extract_features.main(feature_conf, paths['images'], 
                         image_list=references, feature_path=paths['features'])
    
    print("Generating image pairs for matching...")
    pairs_from_exhaustive.main(paths['sfm_pairs'], image_list=references)
    
    print("Matching features between image pairs...")
    match_features.main(matcher_conf, paths['sfm_pairs'], 
                       features=paths['features'], matches=paths['matches'])
    
    print("Running Structure-from-Motion reconstruction...")
    model = reconstruction.main(paths['sfm_dir'], paths['images'], paths['sfm_pairs'], 
                               paths['features'], paths['matches'], image_list=references)
    
    if model is None:
        raise RuntimeError("Failed to reconstruct 3D model. Check your images and parameters.")
    
    print(f"Successfully reconstructed 3D model with:")
    print(f"  - {len(model.images)} registered images")
    print(f"  - {len(model.points3D)} 3D points")
    print(f"  - {len(model.cameras)} camera models")
    
    # Save model information for localization script
    model_info = {
        'sfm_dir': str(paths['sfm_dir']),
        'features_path': str(paths['features']),
        'matches_path': str(paths['matches']),
        'images_path': str(paths['images']),
        'feature_conf': feature_conf,
        'matcher_conf': matcher_conf,
        'references_registered': [model.images[i].name for i in model.reg_image_ids()]
    }
    
    with open(paths['model_info'], 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Model information saved to: {paths['model_info']}")
    
    if visualize:
        print("Visualizing 3D reconstruction...")
        try:
            # 3D model visualization
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', 
                                     name="mapping", points_rgb=True)
            fig.show()
            
            if save_visualization:
                fig.write_html(str(paths['outputs'] / '3d_reconstruction.html'))
                print(f"3D visualization saved to: {paths['outputs'] / '3d_reconstruction.html'}")
            
            # 2D keypoint visualization
            visualization.visualize_sfm_2d(model, paths['images'], 
                                         color_by='visibility', n=2)
            if save_visualization:
                import matplotlib.pyplot as plt
                plt.savefig(paths['outputs'] / 'keypoints_2d.png', dpi=150, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    print("3D mapping completed successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Build 3D map for visual localization")
    parser.add_argument('images', type=str, help='Path to directory containing reference images')
    parser.add_argument('output', type=str, help='Path to save the 3D model and outputs')
    parser.add_argument('--feature-type', type=str, default='disk', 
                       choices=['disk', 'superpoint', 'sift', 'r2d2'],
                       help='Type of features to extract (default: disk)')
    parser.add_argument('--matcher-type', type=str, default='disk+lightglue',
                       choices=['disk+lightglue', 'superglue', 'NN-superpoint', 'NN-ratio'],
                       help='Type of matcher to use (default: disk+lightglue)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization steps')
    parser.add_argument('--no-save-viz', action='store_true',
                       help='Do not save visualization files')
    
    args = parser.parse_args()
    
    try:
        model = build_3d_map(
            images_path=args.images,
            output_path=args.output,
            feature_type=args.feature_type,
            matcher_type=args.matcher_type,
            visualize=not args.no_visualize,
            save_visualization=not args.no_save_viz
        )
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Model saved to: {args.output}")
        print(f"✓ Use the localization script to query images against this model")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
