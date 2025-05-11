#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine all processed datasets into a single file.
"""

import os
import pandas as pd
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_datasets():
    """Combine all processed datasets into a single file."""
    processed_dir = Path("data/processed")
    output_file = processed_dir / "all_combined.csv"
    
    if not processed_dir.exists():
        logger.error(f"Directory not found: {processed_dir}")
        return False
    
    # Find all processed CSV files
    csv_files = list(processed_dir.glob("*_processed.csv"))
    
    if not csv_files:
        logger.warning(f"No processed CSV files found in {processed_dir}")
        return False
    
    logger.info(f"Found {len(csv_files)} processed CSV files")
    
    # Process each file
    dfs = []
    total_rows = 0
    
    for file in csv_files:
        try:
            logger.info(f"Reading {file.name}...")
            df = pd.read_csv(file, encoding='utf-8')
            
            # Add source file column
            df['source_file'] = file.name
            
            # Ensure we have the essential columns
            essential_columns = ['subject', 'body', 'label']
            missing_columns = [col for col in essential_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"File {file.name} is missing essential columns: {missing_columns}")
                # Try to map columns if possible
                if 'subject' not in df.columns and 'title' in df.columns:
                    df['subject'] = df['title']
                if 'body' not in df.columns and 'content' in df.columns:
                    df['body'] = df['content']
                if 'body' not in df.columns and 'text' in df.columns:
                    df['body'] = df['text']
                if 'label' not in df.columns and 'class' in df.columns:
                    df['label'] = df['class']
            
            # Skip files that are still missing essential columns
            missing_columns = [col for col in essential_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Skipping {file.name} due to missing columns: {missing_columns}")
                continue
            
            # Normalize label values
            if 'label' in df.columns:
                # Convert label to lowercase string
                df['label'] = df['label'].astype(str).str.lower()
                
                # Map various label values to standardized ones
                label_mapping = {
                    'spam': 'phishing',
                    'phish': 'phishing',
                    'malicious': 'phishing',
                    'bad': 'phishing',
                    '1': 'phishing',
                    '1.0': 'phishing',
                    'true': 'phishing',
                    
                    'ham': 'legitimate',
                    'legit': 'legitimate',
                    'benign': 'legitimate',
                    'good': 'legitimate',
                    '0': 'legitimate',
                    '0.0': 'legitimate',
                    'false': 'legitimate'
                }
                
                df['label'] = df['label'].map(lambda x: label_mapping.get(x, x))
                
                # Filter to known labels
                valid_labels = ['phishing', 'legitimate']
                mask = df['label'].isin(valid_labels)
                if not mask.all():
                    unknown_labels = df.loc[~mask, 'label'].unique()
                    logger.warning(f"Found unknown labels in {file.name}: {unknown_labels}. These will be set to None.")
                    df.loc[~mask, 'label'] = None
            
            # Add to list of DataFrames
            dfs.append(df)
            total_rows += len(df)
            logger.info(f"Added {len(df)} rows from {file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
    
    if not dfs:
        logger.warning("No valid DataFrames to combine")
        return False
    
    # Combine all DataFrames
    logger.info("Combining all DataFrames...")
    start_time = time.time()
    
    # Find common columns across all DataFrames
    common_columns = set.intersection(*[set(df.columns) for df in dfs])
    logger.info(f"Found {len(common_columns)} common columns across all datasets: {common_columns}")
    
    # If we have the essential columns in common, use only those for a clean merge
    essential_columns = ['subject', 'body', 'label', 'source_file']
    essential_common = [col for col in essential_columns if col in common_columns]
    
    if len(essential_common) >= 3:  # At least subject, body, label
        logger.info(f"Using essential common columns: {essential_common}")
        # Use only essential common columns
        combined_df = pd.concat([df[essential_common] for df in dfs], ignore_index=True)
    else:
        logger.info("Using all columns with outer join")
        # Use all columns with outer join
        combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['subject', 'body'], keep='first')
    if len(combined_df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(combined_df)} duplicate rows")
    
    # Save combined file
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Combined dataset saved to {output_file}")
    logger.info(f"Total rows: {len(combined_df)}")
    
    # Show some statistics
    if 'label' in combined_df.columns:
        phishing_count = len(combined_df[combined_df['label'] == 'phishing'])
        legitimate_count = len(combined_df[combined_df['label'] == 'legitimate'])
        logger.info(f"Phishing emails: {phishing_count}")
        logger.info(f"Legitimate emails: {legitimate_count}")
    
    logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
    return True

if __name__ == "__main__":
    combine_datasets() 