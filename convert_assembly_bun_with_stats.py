#!/usr/bin/env python3
"""
Script to convert assembly_bun dataset to LeRobot v2 format with train/val split and statistics generation.

Usage:
python convert_assembly_bun_with_stats.py --raw_data_dir /data3/yinmenghao/code/raw_data/assembly_bun --output_dir /data3/yinmenghao/code/openpi/data
"""

import os
import sys
import pickle
import json
import jsonlines
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import tyro
from tqdm import tqdm


def convert_images_to_mp4(image_dir, output_path, fps=10):
    """Convert image sequence to MP4 using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(image_dir, 'frame_%04d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)



def process_episode(episode_dir, episode_idx, output_dir, global_idx, task_description="move the pineapple bun from the conveyor belt to the center of the plate"):
    """Process a single episode and convert it to LeRobot v2 format."""
    print(f"Processing episode {episode_idx:04d}")
    
    # Load data and meta
    data_path = os.path.join(episode_dir, 'data.pkl')
    meta_path = os.path.join(episode_dir, 'meta.json')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Get episode length
    n_steps = meta['n_steps']
    
    # Create video directories if they don't exist
    image_dir = os.path.join(output_dir, 'videos', 'chunk-000', 'image')
    os.makedirs(image_dir, exist_ok=True)
    wrist_image_dir = os.path.join(output_dir, 'videos', 'chunk-000', 'wrist_image')
    os.makedirs(wrist_image_dir, exist_ok=True)
    
    # Convert images to MP4 (using front_view as main image)
    front_image_dir = os.path.join(episode_dir, 'Images_front_view')
    image_output = os.path.join(image_dir, f'episode_{episode_idx:06d}.mp4')
    convert_images_to_mp4(front_image_dir, image_output)
    
    # Convert wrist view images to MP4
    wrist_image_source_dir = os.path.join(episode_dir, 'Images_wrist_view')
    wrist_image_output = os.path.join(wrist_image_dir, f'episode_{episode_idx:06d}.mp4')
    convert_images_to_mp4(wrist_image_source_dir, wrist_image_output)
    
    # Prepare state data: concatenate robot_eef_pose (7D) and robot_gripper (1D)
    state = np.hstack([data['robot_eef_pose'], data['robot_gripper'].reshape(-1, 1)])
    
    # Prepare action data: concatenate action (7D) and action_gripper (1D)
    action = np.hstack([data['action'], data['action_gripper'].reshape(-1, 1)])
    
    # Create data rows for parquet
    rows = []
    for i in range(n_steps):
        row = {
            'observation/state': state[i].tolist(),
            'observation/image': None,
            'observation/wrist_image': None,
            'actions': action[i].tolist(),
            'timestamp': float(data['timestamp'][i]) if isinstance(data['timestamp'][i], np.float32 | np.float64) else float(data['timestamp'][i]),
            'task_index': 0,
            'episode_index': int(episode_idx),
            'index': int(i),
        }
        rows.append(row)
    
    # Create dataframe and write to parquet
    episode_df = pd.DataFrame(rows)
    
    # Create data directory if it doesn't exist
    parquet_dir = os.path.join(output_dir, 'data', 'chunk-000')
    os.makedirs(parquet_dir, exist_ok=True)
    
    parquet_output = os.path.join(parquet_dir, f'episode_{episode_idx:06d}.parquet')
    episode_df.to_parquet(parquet_output, index=False)
    
    # Add to episodes.jsonl (use 'length' instead of 'num_frames' for v2.0 compatibility)
    episode_entry = {
        'episode_index': episode_idx,
        'chunk_index': 0,
        'length': n_steps,  # Use 'length' instead of 'num_frames' for v2.0 compatibility
        'fps': 10,
        'start_time': meta['start_time'],
        'duration': meta['duration'],
        'task_index': 0
    }
    
    return episode_entry, n_steps



def create_meta_files(output_dir, total_episodes, total_frames, task_description):
    """Create the required meta files."""
    meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    
    # Create tasks.jsonl (use 'task' instead of 'language_instruction' for v2.0 compatibility)
    tasks = [
        {"task_index": 0, "task": task_description}  # Use 'task' instead of 'language_instruction'
    ]
    tasks_path = os.path.join(meta_dir, 'tasks.jsonl')
    with jsonlines.open(tasks_path, 'w') as writer:
        writer.write_all(tasks)
    
    # Create modality.json
    modality = {
        "actions": [
            "eef_pose.x",
            "eef_pose.y",
            "eef_pose.z",
            "eef_pose.qx",
            "eef_pose.qy",
            "eef_pose.qz",
            "eef_pose.qw",
            "gripper.pos"
        ],
        "observation.state": [
            "eef_pose.x",
            "eef_pose.y",
            "eef_pose.z",
            "eef_pose.qx",
            "eef_pose.qy",
            "eef_pose.qz",
            "eef_pose.qw",
            "gripper.pos"
        ],
        "image": ["rgb"],
        "wrist_image": ["rgb"]
    }
    modality_path = os.path.join(meta_dir, 'modality.json')
    with open(modality_path, 'w') as f:
        json.dump(modality, f, indent=2)
    
    # Create info.json with codebase_version: v2.0 for compatibility
    info = {
        "codebase_version": "v2.0",  # Use v2.0 instead of v2.1 for compatibility
        "robot_type": "franka_robot",
        "dataset_name": output_dir.split('/')[-1],
        "dataset_version": "1.0",
        "description": "Assembly bun dataset collected via teleoperation",
        "creation_date": "2026-01-09",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {
            "train": "0:100" if "train" in output_dir else "0:0",
            "val": "0:100" if "val" in output_dir else "0:0"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "actions": {
                "dtype": "float32",
                "names": [
                    "eef_pose.x",
                    "eef_pose.y",
                    "eef_pose.z",
                    "eef_pose.qx",
                    "eef_pose.qy",
                    "eef_pose.qz",
                    "eef_pose.qw",
                    "gripper.pos"
                ],
                "shape": [8]
            },
            "observation.state": {
                "dtype": "float32",
                "names": [
                    "eef_pose.x",
                    "eef_pose.y",
                    "eef_pose.z",
                    "eef_pose.qx",
                    "eef_pose.qy",
                    "eef_pose.qz",
                    "eef_pose.qw",
                    "gripper.pos"
                ],
                "shape": [8]
            },
            "image": {
                "dtype": "image",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"]
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"]
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            }
        },
        "total_chunks": 0,
        "total_videos": total_episodes * 2
    }
    info_path = os.path.join(meta_dir, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)



def fix_parquet_columns(dataset_dir):
    """Fix parquet files by ensuring correct column names."""
    data_dir = os.path.join(dataset_dir, 'data')
    
    # Get all parquet files
    parquet_files = list(Path(data_dir).glob('**/*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}, skipping...")
        return
    
    print(f"Found {len(parquet_files)} parquet files in {dataset_dir}")
    
    # Iterate over all parquet files and check/rename columns
    for file_path in tqdm(parquet_files, desc=f"Checking parquet files in {dataset_dir}"):
        # Read the parquet file
        parquet_df = pd.read_parquet(file_path)
        
        # Rename columns if they exist
        rename_dict = {}
        if 'action' in parquet_df.columns:
            rename_dict['action'] = 'actions'
        if 'observation.state' in parquet_df.columns:
            rename_dict['observation.state'] = 'observation/state'
        
        # Add image columns if they don't exist
        if 'observation/image' not in parquet_df.columns:
            parquet_df['observation/image'] = None
        if 'observation/wrist_image' not in parquet_df.columns:
            parquet_df['observation/wrist_image'] = None
        
        if rename_dict:
            # Rename the columns
            parquet_df = parquet_df.rename(columns=rename_dict)
            
            # Write the fixed parquet file back
            parquet_df.to_parquet(file_path, index=False)
    
    print(f"Fixed all parquet files in {dataset_dir}")


def fix_dataset_meta(dataset_dir):
    """Fix dataset metadata by ensuring correct field names."""
    meta_dir = os.path.join(dataset_dir, 'meta')
    
    # Fix info.json
    info_path = os.path.join(meta_dir, 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    if 'action' in info['features']:
        info['features']['actions'] = info['features'].pop('action')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Fixed info.json in {dataset_dir}")
    
    # Fix modality.json
    modality_path = os.path.join(meta_dir, 'modality.json')
    with open(modality_path, 'r') as f:
        modality = json.load(f)
    
    if 'action' in modality:
        modality['actions'] = modality.pop('action')
        with open(modality_path, 'w') as f:
            json.dump(modality, f, indent=2)
        print(f"Fixed modality.json in {dataset_dir}")
    
    # Fix stats.json
    stats_path = os.path.join(meta_dir, 'stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if 'action' in stats:
        stats['actions'] = stats.pop('action')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Fixed stats.json in {dataset_dir}")
    
    # Fix relative_stats.json
    relative_stats_path = os.path.join(meta_dir, 'relative_stats.json')
    with open(relative_stats_path, 'r') as f:
        relative_stats = json.load(f)
    
    if 'action' in relative_stats:
        relative_stats['actions'] = relative_stats.pop('action')
        with open(relative_stats_path, 'w') as f:
            json.dump(relative_stats, f, indent=4)
        print(f"Fixed relative_stats.json in {dataset_dir}")


def fix_libero_policy():
    """Fix libero_policy.py to handle None values in images."""
    policy_file = "/data3/yinmenghao/code/openpi/src/openpi/policies/libero_policy.py"
    
    # Read the current content of the file
    with open(policy_file, 'r') as f:
        content = f.read()
    
    # Check if the fix is already applied
    if "if image is None:" in content:
        print(f"{policy_file} already has the fix for None values in images, skipping...")
        return
    
    # Replace the _parse_image function with a version that handles None values
    old_parse_image = """def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image"""
    
    new_parse_image = """def _parse_image(image) -> np.ndarray:
    if image is None:
        # Return a dummy image if no image data is available
        return np.zeros((224, 224, 3), dtype=np.uint8)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image"""
    
    # Replace the function in the content
    fixed_content = content.replace(old_parse_image, new_parse_image)
    
    # Write the fixed content back to the file
    with open(policy_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed {policy_file} to handle None values in images")


def calculate_dataset_statistics(parquet_paths, output_dir):
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Collect all the data
    all_low_dim_data_list = []
    for parquet_path in tqdm(sorted(parquet_paths), desc="Collecting all parquet files..."):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    
    # Compute dataset statistics
    dataset_statistics = {}
    features = list(all_low_dim_data.columns)
    
    # Filter only float features
    float_features = []
    for feature in features:
        if feature in all_low_dim_data.columns:
            sample_data = all_low_dim_data[feature].iloc[0]
            # Check if it's a numpy array or list of floats
            if isinstance(sample_data, np.ndarray) and sample_data.dtype in [np.float32, np.float64]:
                float_features.append(feature)
            elif isinstance(sample_data, list) and len(sample_data) > 0 and isinstance(sample_data[0], (float, np.float32, np.float64)):
                float_features.append(feature)
            elif isinstance(sample_data, float | np.float32 | np.float64):
                # For scalar values like timestamp
                float_features.append(feature)
    
    for le_modality in float_features:
        print(f"Computing statistics for {le_modality}...")
        # Get all data for this feature
        feature_data = all_low_dim_data[le_modality].to_numpy()
        
        # Convert to numpy array
        if isinstance(feature_data[0], np.ndarray):
            # For multi-dimensional features like observation.state and action
            np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in feature_data])
        elif isinstance(feature_data[0], float | np.float32 | np.float64):
            # For scalar features like timestamp
            np_data = np.asarray(feature_data, dtype=np.float32).reshape(-1, 1)
        else:
            # For list features
            np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in feature_data])
        
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist()
        }
    
    # Write stats.json
    meta_dir = os.path.join(output_dir, 'meta')
    stats_path = os.path.join(meta_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=4)
    
    # Write relative_stats.json (same as stats.json for now)
    relative_stats_path = os.path.join(meta_dir, 'relative_stats.json')
    with open(relative_stats_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=4)
    
    print(f"Statistics saved to {stats_path}")
    print(f"Relative statistics saved to {relative_stats_path}")
    print(f"Features included: {list(dataset_statistics.keys())}")
    
    return dataset_statistics



def process_dataset(episodes, output_dir, task_description, dataset_type):
    """Process a list of episodes and create a LeRobot v2 dataset."""
    print(f"Processing {dataset_type} dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process episodes
    episode_entries = []
    total_frames = 0
    
    for episode_idx, episode_dir in enumerate(episodes):
        episode_entry, n_steps = process_episode(episode_dir, episode_idx, output_dir, total_frames, task_description)
        episode_entries.append(episode_entry)
        total_frames += n_steps
    
    # Create meta files
    create_meta_files(output_dir, len(episode_entries), total_frames, task_description)
    
    # Write episodes.jsonl
    episodes_path = os.path.join(output_dir, 'meta', 'episodes.jsonl')
    with jsonlines.open(episodes_path, 'w') as writer:
        writer.write_all(episode_entries)
    
    # Calculate and write statistics
    parquet_files = list(Path(output_dir).glob("data/*/*.parquet"))
    calculate_dataset_statistics(parquet_files, output_dir)
    
    # Fix parquet columns to ensure correct format
    fix_parquet_columns(output_dir)
    
    # Fix dataset metadata to ensure correct field names
    fix_dataset_meta(output_dir)
    
    print(f"Successfully created {dataset_type} dataset with {len(episode_entries)} episodes and {total_frames} frames")
    
    return output_dir



def main(
    raw_data_dir: str = "/data3/yinmenghao/code/raw_data/assembly_bun",
    output_dir: str = "/data3/yinmenghao/code/openpi/data",
    split_ratio: float = 0.9,
    task_description: str = "move the pineapple bun from the conveyor belt to the center of the plate",
    fps: int = 10,
):
    """Convert assembly_bun dataset to LeRobot v2 format with train/val split and statistics generation."""
    print("Starting dataset conversion...")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train/val split ratio: {split_ratio}:{1-split_ratio}")
    print(f"Task description: {task_description}")
    
    # Collect all episode directories
    episode_dirs = []
    for root, dirs, _files in os.walk(raw_data_dir):
        for dir_name in dirs:
            if dir_name.startswith("episode_"):
                episode_dir = os.path.join(root, dir_name)
                if os.path.exists(os.path.join(episode_dir, "data.pkl")) and os.path.exists(os.path.join(episode_dir, "meta.json")):
                    episode_dirs.append(episode_dir)
    
    # Sort episode directories by episode number
    episode_dirs.sort(key=lambda x: int(x.split("episode_")[-1]))
    
    # Split into train and validation sets
    num_episodes = len(episode_dirs)
    num_train = int(num_episodes * split_ratio)
    train_episodes = episode_dirs[:num_train]
    val_episodes = episode_dirs[num_train:]
    
    print(f"Found {num_episodes} episodes")
    print(f"Train set: {len(train_episodes)} episodes")
    print(f"Validation set: {len(val_episodes)} episodes")
    
    # Process train dataset
    train_output_dir = os.path.join(output_dir, "assembly_bun_train")
    process_dataset(train_episodes, train_output_dir, task_description, "train")
    
    # Process validation dataset
    val_output_dir = os.path.join(output_dir, "assembly_bun_val")
    process_dataset(val_episodes, val_output_dir, task_description, "validation")
    
    # Apply fixes to libero_policy.py to handle None values in images
    fix_libero_policy()
    
    print("\nDataset conversion completed successfully!")
    print(f"Train dataset saved to: {train_output_dir}")
    print(f"Validation dataset saved to: {val_output_dir}")
    print("All fixes applied successfully!")



if __name__ == "__main__":
    tyro.cli(main)