#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process individual files from the phishing email dataset.
"""

import os
import sys
import pandas as pd
import csv
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(file_path, output_dir):
    """Process a single file and save the output."""
    try:
        logger.info(f"Processing {file_path.name}...")
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{file_path.stem}_processed.csv"
        
        # Process based on file type
        if file_path.suffix.lower() == '.csv':
            # Try to read the CSV file with different encodings
            for encoding in ['latin-1', 'utf-8', 'cp1252']:
                try:
                    logger.info(f"Trying to read {file_path.name} with {encoding} encoding")
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', 
                                    quoting=3, low_memory=False)
                    if not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"Failed to read with {encoding}: {str(e)}")
            
            # If standard approaches fail, try with Python's csv module
            if 'df' not in locals() or df.empty:
                logger.info(f"Trying with Python's csv module for {file_path.name}")
                rows = []
                with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                    # Try to detect the dialect
                    sample = f.read(4096)
                    f.seek(0)
                    
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        reader = csv.reader(f, dialect)
                    except:
                        # If dialect detection fails, use default CSV reader
                        reader = csv.reader(f)
                    
                    # Read in chunks to avoid memory issues
                    chunk_size = 10000
                    chunk = []
                    for i, row in enumerate(reader):
                        if i == 0:  # First row is header
                            headers = row
                            continue
                            
                        if row:  # Skip empty rows
                            chunk.append(row)
                            
                        if len(chunk) >= chunk_size:
                            # Process and save this chunk
                            chunk_df = pd.DataFrame(chunk, columns=headers)
                            if i == chunk_size:  # First chunk
                                chunk_df.to_csv(output_file, index=False, encoding='utf-8', mode='w')
                            else:
                                chunk_df.to_csv(output_file, index=False, encoding='utf-8', 
                                              mode='a', header=False)
                            chunk = []
                    
                    # Process any remaining rows
                    if chunk:
                        chunk_df = pd.DataFrame(chunk, columns=headers)
                        if 'df' not in locals():  # No previous chunks
                            chunk_df.to_csv(output_file, index=False, encoding='utf-8', mode='w')
                        else:
                            chunk_df.to_csv(output_file, index=False, encoding='utf-8', 
                                          mode='a', header=False)
                
                logger.info(f"File saved to {output_file}")
                logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
                return True
            
            # Save the DataFrame
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"File saved to {output_file} with {len(df)} rows")
            logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
            return True
            
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
        return False

def main():
    """Process files from the phishing email dataset."""
    # Path to the raw data
    raw_data_path = Path("data/raw/phishing-email-dataset")
    
    # Output path for processed data
    processed_data_path = Path("data/processed")
    
    # Get list of files
    files = list(raw_data_path.glob("*.csv"))
    
    if not files:
        logger.warning(f"No files found in {raw_data_path}")
        return
    
    logger.info(f"Found {len(files)} files in {raw_data_path}")
    
    # Process each file
    for file in files:
        process_file(file, processed_data_path)

if __name__ == "__main__":
    main() 