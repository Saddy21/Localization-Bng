#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Image Localization Script
This script localizes a query image within a pre-built 3D map.
It estimates the 6DOF camera pose (position and orientation) of the query image.
"""

import argparse
import os
import sys
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# Import HLoc and COLMAP modules
from hloc import extract_features, match_features, visualization, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils import viz_3d
import pycolmap


def load_model_info(model_info_path):
    """Load the model information saved during training."""
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info file not found: {model_info_path}")
    
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    return model_info


def load_3d_model(sfm_dir):
    """Load the 3D reconstruction model."""
    sfm_path = Path(sfm_dir)
    if not sfm_path.exists():
        raise FileNotFoundError(f"SfM directory not found: {sfm_dir}")
    
    model = pycolmap.Reconstruction(sfm_path)
    return model


def localize_image(query_image_path, model_info_path, output_dir=None, 
                  visualize=True, save_results=True, cleanup_query=True,
                  min_inliers=12, max_ransac_error=12.0):
    """
    Localize a query image within the pre-built 3D map.
    
    Args:
        query_image_path: Path to the query image
        model_info_path: Path to the model info file from training
        output_dir: Directory to save localization results
        visualize: Whether to show visualizations
        save_results: Whether to save result files
        cleanup_query: Whether to remove copied query image after processing
        min_inliers: Minimum number of inliers for reliable pose
        max_ransac_error: Maximum RANSAC error for pose estimation
    
    Returns:
        dict: Localization results containing pose, inliers, etc.
    """
    
    print("Loading model information...")
    model_info = load_model_info(model_info_path)
    
    print("Loading 3D reconstruction model...")
    model = load_3d_model(model_info['sfm_dir'])
    
    print(f"Model loaded with:")
    print(f"  - {len(model.images)} registered images")
    print(f"  - {len(model.points3D)} 3D points")
    
    # Setup paths
    query_path = Path(query_image_path)
    original_query_path = query_path  # Keep reference to original path
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    
    images_path = Path(model_info['images_path'])
    features_path = Path(model_info['features_path'])
    matches_path = Path(model_info['matches_path'])
    
    if output_dir is None:
        output_dir = query_path.parent / 'localization_results'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define localization-specific files
    loc_pairs = output_path / 'pairs-loc.txt'
    
    # Handle query path - copy query image to images directory if needed
    query_was_copied = False
    query_dest = None
    try:
        query_relative = str(query_path.relative_to(images_path))
        query_dest = query_path  # Already in the correct location
    except ValueError:
        # Query image is not in the images directory, copy it there
        import shutil
        query_filename = query_path.name
        query_dest = images_path / query_filename
        
        # Ensure unique filename if it already exists
        counter = 1
        while query_dest.exists():
            name_parts = query_path.stem, counter, query_path.suffix
            query_dest = images_path / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            counter += 1
        
        print(f"Copying query image to: {query_dest}")
        shutil.copy2(query_path, query_dest)
        query_relative = str(query_dest.relative_to(images_path))
        query_was_copied = True
    
    print(f"Localizing query image: {query_relative}")
    
    # Display query image
    if visualize:
        print("Displaying query image...")
        try:
            img = Image.open(original_query_path)  # Use original path for display
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Query Image: {original_query_path.name}', fontsize=14, pad=20)
            plt.tight_layout()
            if save_results:
                plt.savefig(output_path / 'query_image.png', dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display query image: {e}")
    
    print("Extracting features from query image...")
    extract_features.main(model_info['feature_conf'], images_path, 
                         image_list=[query_relative], feature_path=features_path, 
                         overwrite=True)
    
    print("Generating query-reference image pairs...")
    references_registered = model_info['references_registered']
    pairs_from_exhaustive.main(loc_pairs, image_list=[query_relative], 
                              ref_list=references_registered)
    
    print("Matching query features with reference features...")
    match_features.main(model_info['matcher_conf'], loc_pairs, 
                       features=features_path, matches=matches_path, 
                       overwrite=True)
    
    print("Inferring camera parameters from EXIF data...")
    camera = pycolmap.infer_camera_from_image(query_dest)
    
    # Handle different pycolmap versions - some have model_name, others have model
    try:
        model_name = camera.model_name
    except AttributeError:
        # Fallback for older versions
        model_name = getattr(camera, 'model', 'Unknown')
        if hasattr(pycolmap, 'CameraModelNameToId'):
            # Convert model ID to name if possible
            model_id_to_name = {v: k for k, v in pycolmap.CameraModelNameToId().items()}
            if isinstance(model_name, int) and model_name in model_id_to_name:
                model_name = model_id_to_name[model_name]
    
    print(f"Camera model: {model_name}")
    print(f"Image dimensions: {camera.width} x {camera.height}")
    
    # Handle camera parameters safely
    if len(camera.params) > 0:
        print(f"Focal length: {camera.params[0]:.2f} pixels")
        if len(camera.params) > 2:
            print(f"Principal point: ({camera.params[1]:.2f}, {camera.params[2]:.2f})")
        if len(camera.params) > 3:
            print(color_green(f"Distortion parameter: {camera.params[3]:.6f}"))
    else:
        print("Focal length: Not available")
    
    print("Estimating camera pose...")
    ref_ids = [model.find_image_with_name(n).image_id for n in references_registered]
    
    # Localization configuration with customizable parameters
    conf = {
        'estimation': {'ransac': {'max_error': max_ransac_error}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    
    localizer = QueryLocalizer(model, conf)
    ret, log = pose_from_cluster(localizer, query_relative, camera, ref_ids, 
                                features_path, matches_path)
    
    if ret is None:
        print("❌ Localization failed - could not estimate camera pose")
        if query_was_copied and cleanup_query and query_dest:
            try:
                query_dest.unlink()
                print(f"Cleaned up copied query image: {query_dest}")
            except Exception as e:
                print(f"Warning: Could not remove copied query image: {e}")
        return None
    
    num_inliers = ret['num_inliers']
    total_matches = len(ret['inliers']) if 'inliers' in ret else 0
    success = num_inliers >= min_inliers  # Configurable minimum number of inliers
    
    print(f"\n{'✓' if success else '⚠'} Localization {'succeeded' if success else 'uncertain'}:")
    print(f"  - Inlier correspondences: {num_inliers}/{total_matches}")
    if total_matches > 0:
        inlier_ratio = num_inliers / total_matches
        print(f"  - Inlier ratio: {inlier_ratio*100:.1f}%")
    else:
        inlier_ratio = 0.0
        print(f"  - Inlier ratio: N/A (no matches)")
    
    # Extract pose information
    camera_pose = ret['cam_from_world']
    camera_position = camera_pose.translation
    camera_rotation = camera_pose.rotation.matrix()
    
    print(f"  - Camera position (x, y, z): [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
    print(f"  - Camera orientation (quaternion w, x, y, z): [{camera_pose.rotation.quat[0]:.6f}, {camera_pose.rotation.quat[1]:.6f}, {camera_pose.rotation.quat[2]:.6f}, {camera_pose.rotation.quat[3]:.6f}]")
    
    # Calculate additional metrics
    reprojection_error = log.get('avg_reproj_error', None) if log else None
    
    # Prepare results
    results = {
        'success': success,
        'num_inliers': num_inliers,
        'total_matches': total_matches,
        'inlier_ratio': inlier_ratio,
        'camera_position': camera_position.tolist(),
        'camera_rotation_matrix': camera_rotation.tolist(),
        'camera_quaternion': camera_pose.rotation.quat.tolist(),
        'cam_from_world': camera_pose,
        'camera_params': {
            'model': str(model_name),
            'width': camera.width,
            'height': camera.height,
            'params': camera.params.tolist()
        },
        'query_path': str(original_query_path),
        'reprojection_error': reprojection_error,
        'config': {
            'min_inliers': min_inliers,
            'max_ransac_error': max_ransac_error
        },
        'log': log
    }
    
    if save_results:
        # Save results to file
        results_file = output_path / 'localization_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"  - Results saved to: {results_file}")
        
        # Save pose as text file
        pose_file = output_path / 'camera_pose.txt'
        with open(pose_file, 'w') as f:
            f.write(f"Query Image: {original_query_path.name}\n")
            f.write(f"Success: {success}\n")
            f.write(f"Inliers: {num_inliers}/{total_matches}")
            if total_matches > 0:
                f.write(f" ({inlier_ratio*100:.1f}%)")
            f.write(f"\n")
            if reprojection_error is not None:
                f.write(f"Average Reprojection Error: {reprojection_error:.3f} pixels\n")
            f.write(f"\nCamera Position (x, y, z):\n")
            f.write(f"{camera_position[0]:.6f} {camera_position[1]:.6f} {camera_position[2]:.6f}\n\n")
            f.write(f"Camera Orientation (quaternion w, x, y, z):\n")
            quat = camera_pose.rotation.quat
            f.write(f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n\n")
            f.write(f"Camera Rotation Matrix:\n")
            for row in camera_rotation:
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        print(f"  - Pose saved to: {pose_file}")
        
        # Save JSON format for easier programmatic access
        json_results = {k: v for k, v in results.items() 
                       if k not in ['cam_from_world', 'log']}  # Exclude non-serializable objects
        json_file = output_path / 'localization_results.json'
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"  - JSON results saved to: {json_file}")
    
    # Create visualizations BEFORE cleanup
    if visualize and success:
        print("Creating visualizations...")
        try:
            # Visualize 2D correspondences
            print("Visualizing 2D correspondences...")
            fig_2d = visualization.visualize_loc_from_log(images_path, query_relative, log, model)
            if save_results:
                plt.savefig(output_path / 'correspondences_2d.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Create 3D visualization
            print("Creating 3D visualization with camera pose...")
            
            # Initialize the figure
            fig = viz_3d.init_figure()
            
            # Plot the 3D reconstruction (red points and cameras)
            viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', 
                                     name="mapping", points_rgb=True)
            
            # Plot the localized camera (green camera)
            pose = pycolmap.Image(cam_from_world=ret['cam_from_world'])
            viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.8)', 
                                    name=f"Query: {query_relative}", fill=True)
            
            # Plot inlier 3D points (lime colored points)
            if 'inliers' in ret and 'points3D_ids' in log:
                inl_3d = np.array([model.points3D[pid].xyz for pid in 
                                  np.array(log['points3D_ids'])[ret['inliers']]])
                viz_3d.plot_points(fig, inl_3d, color="lime", ps=2, 
                                  name=f"{query_relative}_inliers")
            
            # Improve the layout
            fig.update_layout(
                title=f"3D Localization Result: {original_query_path.name}",
                scene=dict(
                    aspectmode="data",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            )
            
            # Show the interactive plot
            fig.show()
            
            if save_results:
                fig.write_html(str(output_path / 'localization_3d.html'))
                print(f"  - 3D visualization saved to: {output_path / 'localization_3d.html'}")
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup: remove copied query image AFTER visualizations
    if query_was_copied and cleanup_query and query_dest:
        try:
            query_dest.unlink()
            print(f"Cleaned up copied query image: {query_dest}")
        except Exception as e:
            print(f"Warning: Could not remove copied query image: {e}")
    
    # Print final camera position
    print(f"\nFinal Camera Position: [{camera_position[0]:.6f}, {camera_position[1]:.6f}, {camera_position[2]:.6f}]")
    
    return results


def color_green(text):
    """Return text in green color for terminal output."""
    return f"\033[92m{text}\033[0m"


def batch_localize(query_dir, model_info_path, output_dir=None, **kwargs):
    """
    Localize multiple images in a directory.
    
    Args:
        query_dir: Directory containing query images
        model_info_path: Path to model_info.pkl from training
        output_dir: Base output directory for results
        **kwargs: Additional arguments passed to localize_image
    
    Returns:
        dict: Results for all processed images
    """
    query_path = Path(query_dir)
    if not query_path.exists():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in query_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {query_dir}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = {}
    successful_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        print(f"{'='*60}")
        
        # Create individual output directory for each image
        if output_dir:
            img_output_dir = Path(output_dir) / image_file.stem
        else:
            img_output_dir = image_file.parent / f"{image_file.stem}_localization"
        
        try:
            result = localize_image(
                query_image_path=str(image_file),
                model_info_path=model_info_path,
                output_dir=str(img_output_dir),
                **kwargs
            )
            
            if result and result['success']:
                successful_count += 1
                all_results[str(image_file)] = result
                print(f"✓ Successfully localized {image_file.name}")
            else:
                all_results[str(image_file)] = None
                print(f"❌ Failed to localize {image_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}")
            all_results[str(image_file)] = None
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete: {successful_count}/{len(image_files)} images localized successfully")
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Localize image(s) in pre-built 3D map")
    parser.add_argument('query', type=str, help='Path to query image or directory')
    parser.add_argument('model_info', type=str, help='Path to model_info.pkl from training')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization steps')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save result files')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Do not remove copied query image (if copied)')
    parser.add_argument('--min-inliers', type=int, default=12,
                       help='Minimum number of inliers for reliable pose (default: 12)')
    parser.add_argument('--max-error', type=float, default=12.0,
                       help='Maximum RANSAC error for pose estimation (default: 12.0)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in the query directory')
    
    args = parser.parse_args()
    
    try:
        query_path = Path(args.query)
        
        if args.batch or query_path.is_dir():
            # Batch processing
            results = batch_localize(
                query_dir=args.query,
                model_info_path=args.model_info,
                output_dir=args.output,
                visualize=not args.no_visualize,
                save_results=not args.no_save,
                cleanup_query=not args.no_cleanup,
                min_inliers=args.min_inliers,
                max_ransac_error=args.max_error
            )
            
            successful_results = [r for r in results.values() if r and r['success']]
            if successful_results:
                print(f"\n✓ Batch localization completed successfully!")
                print(f"✓ {len(successful_results)} out of {len(results)} images localized")
            else:
                print(f"\n❌ Batch localization failed for all images")
                sys.exit(1)
        else:
            # Single image processing
            result = localize_image(
                query_image_path=args.query,
                model_info_path=args.model_info,
                output_dir=args.output,
                visualize=not args.no_visualize,
                save_results=not args.no_save,
                cleanup_query=not args.no_cleanup,
                min_inliers=args.min_inliers,
                max_ransac_error=args.max_error
            )
            
            if result is None:
                print("\n❌ Localization failed")
                sys.exit(1)
            elif result['success']:
                print(f"\n✓ Localization completed successfully!")
                print(f"✓ Camera pose estimated with {result['inlier_ratio']*100:.1f}% inlier ratio")
            else:
                print(f"\n⚠ Localization completed with low confidence")
                print(f"⚠ Only {result['num_inliers']} inliers found - results may be unreliable")
        
    except Exception as e:
        print(f"Error during localization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
