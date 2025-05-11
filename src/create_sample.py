#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a small sample dataset for testing purposes.
"""

import pandas as pd
from pathlib import Path
import logging
import traceback
from tqdm import tqdm
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample(input_path, output_path, sample_size=1000, random_state=42, balanced=True):
    """
    Create a small sample dataset from a larger one.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the output CSV file
        sample_size: Number of rows to sample
        random_state: Random seed for reproducibility
        balanced: Whether to create a balanced sample (equal number of phishing and legitimate)
    """
    try:
        start_time = time.time()
        logger.info(f"Loading data from {input_path}")
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Show progress while reading large CSV file
        with tqdm(total=1, desc="Loading data") as pbar:
            df = pd.read_csv(input_path)
            pbar.update(1)
            
        logger.info(f"Original dataset size: {len(df)} rows")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        # Filter to only use rows with known labels
        with tqdm(total=1, desc="Filtering data") as pbar:
            df = df[df['label'].isin(['phishing', 'legitimate'])].reset_index(drop=True)
            pbar.update(1)
            
        logger.info(f"Filtered dataset size: {len(df)} rows")
        logger.debug(f"Original class distribution: {df['label'].value_counts().to_dict()}")
        
        # Take a stratified sample
        logger.info(f"Taking {'balanced ' if balanced else ''}sample of {sample_size} rows")
        
        phishing_count = len(df[df['label'] == 'phishing'])
        legitimate_count = len(df[df['label'] == 'legitimate'])
        
        logger.debug(f"Available phishing emails: {phishing_count}")
        logger.debug(f"Available legitimate emails: {legitimate_count}")
        
        if balanced:
            # Equal number of phishing and legitimate emails
            phishing_sample_size = min(sample_size // 2, phishing_count)
            legitimate_sample_size = min(sample_size // 2, legitimate_count)
        else:
            # Maintain original class distribution
            total_count = phishing_count + legitimate_count
            phishing_sample_size = min(int(sample_size * phishing_count / total_count), phishing_count)
            legitimate_sample_size = min(int(sample_size * legitimate_count / total_count), legitimate_count)
        
        logger.debug(f"Sampling {phishing_sample_size} phishing emails")
        logger.debug(f"Sampling {legitimate_sample_size} legitimate emails")
        
        with tqdm(total=2, desc="Sampling") as pbar:
            phishing = df[df['label'] == 'phishing'].sample(
                phishing_sample_size, 
                random_state=random_state
            )
            pbar.update(1)
            
            legitimate = df[df['label'] == 'legitimate'].sample(
                legitimate_sample_size, 
                random_state=random_state
            )
            pbar.update(1)
        
        # Combine samples
        with tqdm(total=1, desc="Combining samples") as pbar:
            sample = pd.concat([phishing, legitimate]).reset_index(drop=True)
            pbar.update(1)
            
        logger.info(f"Sample size: {len(sample)} rows")
        logger.info(f"Class distribution: {sample['label'].value_counts().to_dict()}")
        
        # Save the sample
        logger.info(f"Saving sample to {output_path}")
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with tqdm(total=1, desc="Saving data") as pbar:
            sample.to_csv(output_path, index=False)
            pbar.update(1)
            
        logger.info(f"Sample saved to {output_path}")
        
        end_time = time.time()
        logger.info(f"Sample creation completed in {end_time - start_time:.2f} seconds")
        
        return sample
        
    except Exception as e:
        logger.error(f"Error creating sample: {e}")
        logger.debug(traceback.format_exc())
        raise

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a sample dataset for testing")
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/all_combined.csv",
        help="Path to the input CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/test_sample.csv",
        help="Path to save the output CSV file"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Number of rows to sample"
    )
    
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Create a balanced sample (equal number of phishing and legitimate)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        logger.info("Starting sample creation process")
        
        args = parse_args()
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        create_sample(input_path, output_path, sample_size=args.size, balanced=args.balanced)
        
        logger.info("Sample creation process completed")
        
    except Exception as e:
        logger.error(f"Error in sample creation process: {e}")
        logger.debug(traceback.format_exc()) 