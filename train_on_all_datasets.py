#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the phishing detection model on all datasets in the data/processed directory.
"""

import os
import pandas as pd
import glob
import logging
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import gc
import sys

# Import the training function from train_mlp_sklearn.py
from train_mlp_sklearn import main as train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("all_datasets_training.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

def combine_datasets(processed_dir, output_file=None, exclude_files=None):
    """
    Combine all CSV datasets in the processed directory.
    
    Args:
        processed_dir: Directory containing the processed datasets
        output_file: Path to save the combined dataset (optional)
        exclude_files: List of filenames to exclude (optional)
        
    Returns:
        Path to the combined dataset file
    """
    if exclude_files is None:
        exclude_files = ['all_combined.csv', 'test_sample.csv', '.gitkeep']
    else:
        exclude_files = exclude_files + ['all_combined.csv', 'test_sample.csv', '.gitkeep']
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    # Filter out excluded files
    csv_files = [f for f in csv_files if os.path.basename(f) not in exclude_files]
    
    if not csv_files:
        logger.error(f"No CSV files found in {processed_dir}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files to combine")
    
    # Create a list to store dataframes
    dfs = []
    
    # Read each CSV file
    for file in tqdm(csv_files, desc="Reading datasets"):
        try:
            logger.info(f"Reading {os.path.basename(file)}")
            df = pd.read_csv(file)
            
            # Check if the file has the required columns
            if 'subject' not in df.columns or 'body' not in df.columns or 'label' not in df.columns:
                logger.warning(f"Skipping {file} due to missing required columns")
                continue
                
            # Add source column
            df['source'] = os.path.basename(file)
            
            # Append to list
            dfs.append(df)
            
            # Log the number of rows
            logger.info(f"{os.path.basename(file)}: {len(df)} rows")
            
            # Free up memory
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    if not dfs:
        logger.error("No valid datasets found")
        return None
    
    # Combine all dataframes
    logger.info("Combining datasets")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Log the total number of rows
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    
    # Save the combined dataset if output_file is provided
    if output_file:
        logger.info(f"Saving combined dataset to {output_file}")
        combined_df.to_csv(output_file, index=False)
        return output_file
    
    return combined_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train on all datasets")
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="../data/processed",
        help="Directory containing processed datasets"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="../data/processed/combined_all.csv",
        help="Path to save the combined dataset"
    )
    
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        help="Files to exclude from combination"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=7500,
        help="Maximum number of TF-IDF features to use"
    )
    
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use"
    )
    
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip dataset combination and use existing combined file"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Combine datasets or use existing combined file
    if args.skip_combine:
        logger.info(f"Skipping dataset combination, using existing file: {args.output_file}")
        combined_file = args.output_file
    else:
        logger.info("Combining all datasets")
        combined_file = combine_datasets(args.processed_dir, args.output_file, args.exclude)
    
    if not combined_file or not os.path.exists(combined_file):
        logger.error("Combined dataset not found")
        return 1
    
    # Train the model on the combined dataset
    logger.info(f"Training model on combined dataset: {combined_file}")
    
    # Create a temporary symlink to the combined file as all_combined.csv
    temp_link = os.path.join(os.path.dirname(combined_file), "all_combined.csv")
    
    try:
        # Train the model
        train_model(use_sample=False, max_features=args.max_features)
        
        # Calculate and log total time
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        print("\n" + "=" * 80)
        print("TRAINING ON ALL DATASETS COMPLETED")
        print("=" * 80)
        print(f"Total time: {total_time/60:.2f} minutes")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 