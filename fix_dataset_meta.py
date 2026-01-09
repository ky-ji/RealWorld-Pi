#!/usr/bin/env python3
"""
Script to fix dataset metadata by changing 'action' to 'actions' in meta files.
"""

import os
import json
import jsonlines
from pathlib import Path

def fix_dataset_meta(dataset_dir):
    """Fix dataset metadata by changing 'action' to 'actions' in meta files."""
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

def main():
    """Main function to fix all dataset metadata files."""
    datasets = [
        '/data3/yinmenghao/code/openpi/data/assembly_bun_train',
        '/data3/yinmenghao/code/openpi/data/assembly_bun_val'
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            fix_dataset_meta(dataset)
        else:
            print(f"Dataset {dataset} does not exist, skipping...")
    
    print("All datasets fixed successfully!")

if __name__ == "__main__":
    main()
