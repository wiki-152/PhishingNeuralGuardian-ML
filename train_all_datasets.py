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
    
    # Ensure processed_dir is a Path object
    processed_dir = Path(processed_dir)
    
    # Check if directory exists
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return None
        
    # Get all CSV files in the directory
    csv_files = list(processed_dir.glob("*.csv"))
    
    # Debug
    logger.info(f"Looking for CSV files in: {processed_dir.absolute()}")
    
    # Filter out excluded files
    csv_files = [f for f in csv_files if f.name not in exclude_files]
    
    if not csv_files:
        logger.error(f"No CSV files found in {processed_dir}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files to combine")
    for file in csv_files:
        logger.info(f"  - {file.name}")
    
    # Create a list to store dataframes
    dfs = []
    
    # Read each CSV file
    for file in tqdm(csv_files, desc="Reading datasets"):
        try:
            logger.info(f"Reading {file.name}")
            df = pd.read_csv(file)
            
            # Check if the file has the required columns
            if 'subject' not in df.columns or 'body' not in df.columns or 'label' not in df.columns:
                logger.warning(f"Skipping {file} due to missing required columns")
                continue
                
            # Add source column
            df['source'] = file.name
            
            # Append to list
            dfs.append(df)
            
            # Log the number of rows
            logger.info(f"{file.name}: {len(df)} rows")
            
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
        output_path = Path(output_file)
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving combined dataset to {output_path}")
        combined_df.to_csv(output_path, index=False)
        return str(output_path)
    
    return combined_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train on all datasets")
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed datasets"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/all_combined.csv",
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
    
    if not combined_file:
        logger.error("Combined dataset not created")
        return 1
        
    output_path = Path(combined_file)
    if not output_path.exists():
        logger.error(f"Combined dataset file not found: {output_path}")
        return 1
    
    # Train the model on the combined dataset
    logger.info(f"Training model on combined dataset: {combined_file}")
    
    # Import the training function directly
    try:
        # Add src directory to path if needed
        if not os.path.exists('src'):
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            
        # Import the training function from train_mlp_sklearn.py
        from src.train_mlp_sklearn import main as train_model
        
        # Train the model
        logger.info("Starting model training...")
        train_model(use_sample=False, max_features=args.max_features, data_path=combined_file)
        
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
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 