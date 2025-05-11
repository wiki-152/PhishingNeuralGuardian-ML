#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process just the phishing_email.csv file.
"""

import os
import pandas as pd
import csv
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_phishing_email_csv():
    """Process the phishing_email.csv file using a line-by-line approach."""
    file_path = Path("data/raw/phishing-email-dataset/phishing_email.csv")
    output_path = Path("data/processed/phishing_email_processed.csv")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"Processing {file_path.name}...")
    start_time = time.time()
    
    try:
        # Read the first few lines to determine the header
        with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
            header_line = f.readline().strip()
        
        # Parse the header
        header = header_line.split(',')
        logger.info(f"Header has {len(header)} columns: {header}")
        
        # Map header columns to our required columns
        header_mapping = {}
        for i, col in enumerate(header):
            col_lower = col.lower()
            if 'subject' in col_lower or 'title' in col_lower:
                header_mapping['subject'] = i
            elif 'body' in col_lower or 'content' in col_lower or 'text' in col_lower or 'message' in col_lower:
                header_mapping['body'] = i
            elif 'label' in col_lower or 'class' in col_lower or 'type' in col_lower or 'spam' in col_lower:
                header_mapping['label'] = i
        
        # Check if we found the required columns
        required_cols = ['subject', 'body', 'label']
        missing_cols = [col for col in required_cols if col not in header_mapping]
        
        if missing_cols:
            logger.warning(f"Could not find columns for: {missing_cols}")
            # Add missing columns to header
            for col in missing_cols:
                header.append(col)
        
        # Create a new header with our required columns first
        new_header = ['subject', 'body', 'label'] + [col for col in header if col.lower() not in ['subject', 'body', 'label']]
        
        # Process the file line by line and write directly to output
        with open(file_path, 'r', encoding='latin-1', errors='replace') as infile, \
             open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            
            # Write the new header
            writer = csv.writer(outfile)
            writer.writerow(new_header)
            
            # Skip the header line in the input file
            next(infile)
            
            # Process line by line
            line_count = 0
            error_count = 0
            
            for line in infile:
                line_count += 1
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count} lines...")
                
                try:
                    # Try to parse the line as CSV
                    row = next(csv.reader([line]))
                    
                    # Make sure the row has enough columns for the original header
                    if len(row) < len(header):
                        # Pad with empty strings
                        row.extend([''] * (len(header) - len(row)))
                    
                    # Create a new row with our required columns first
                    new_row = []
                    
                    # Add subject
                    if 'subject' in header_mapping:
                        new_row.append(row[header_mapping['subject']])
                    else:
                        new_row.append('')  # Empty subject
                    
                    # Add body
                    if 'body' in header_mapping:
                        new_row.append(row[header_mapping['body']])
                    else:
                        # Try to use the longest text field as body
                        text_cols = [i for i, val in enumerate(row) if len(val) > 100]
                        if text_cols:
                            longest_text_col = max(text_cols, key=lambda i: len(row[i]))
                            new_row.append(row[longest_text_col])
                        else:
                            new_row.append('')  # Empty body
                    
                    # Add label
                    if 'label' in header_mapping:
                        label_val = row[header_mapping['label']]
                        # Normalize label
                        if label_val.lower() in ['1', 'spam', 'phish', 'phishing']:
                            new_row.append('phishing')
                        elif label_val.lower() in ['0', 'ham', 'legitimate', 'legit']:
                            new_row.append('legitimate')
                        else:
                            new_row.append(label_val)
                    else:
                        new_row.append('unknown')  # Unknown label
                    
                    # Add remaining columns
                    for i, val in enumerate(row):
                        if i not in header_mapping.values():
                            new_row.append(val)
                    
                    # Write the new row
                    writer.writerow(new_row)
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:  # Only log the first 10 errors
                        logger.warning(f"Error processing line {line_count}: {str(e)}")
                    elif error_count == 11:
                        logger.warning("Too many errors, suppressing further error messages")
        
        logger.info(f"Processing complete. Processed {line_count} lines with {error_count} errors.")
        logger.info(f"Output saved to {output_path}")
        logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    process_phishing_email_csv()