#!/usr/bin/env python3
"""
Script to fix parquet files by renaming columns.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def fix_parquet_columns(dataset_dir):
    """Fix parquet files by renaming columns."""
    data_dir = os.path.join(dataset_dir, 'data')
    
    # Get all parquet files
    parquet_files = list(Path(data_dir).glob('**/*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_dir}, skipping...")
        return
    
    print(f"Found {len(parquet_files)} parquet files in {dataset_dir}")
    
    # Iterate over all parquet files and rename columns
    for file_path in tqdm(parquet_files, desc=f"Fixing parquet files in {dataset_dir}"):
        # Read the parquet file
        parquet_df = pd.read_parquet(file_path)
        
        # Rename columns if they exist
        rename_dict = {}
        if 'action' in parquet_df.columns:
            rename_dict['action'] = 'actions'
        if 'observation.state' in parquet_df.columns:
            rename_dict['observation.state'] = 'observation/state'
        if 'observation/image' not in parquet_df.columns:
            # Add empty image columns if they don't exist
            parquet_df['observation/image'] = None
            parquet_df['observation/wrist_image'] = None
        
        if rename_dict:
            # Rename the columns
            parquet_df = parquet_df.rename(columns=rename_dict)
            
            # Write the fixed parquet file back
            parquet_df.to_parquet(file_path, index=False)
    
    print(f"Fixed all parquet files in {dataset_dir}")

def main():
    """Main function to fix all parquet files."""
    datasets = [
        '/data3/yinmenghao/code/openpi/data/assembly_bun_train',
        '/data3/yinmenghao/code/openpi/data/assembly_bun_val'
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            fix_parquet_columns(dataset)
        else:
            print(f"Dataset {dataset} does not exist, skipping...")
    
    print("All parquet files fixed successfully!")

if __name__ == "__main__":
    main()
