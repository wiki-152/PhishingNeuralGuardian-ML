#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process the phishing email dataset using the data_loader module.
"""

import os
import sys
import pandas as pd
import csv
from pathlib import Path
import logging
import time
from tqdm import tqdm

# Add the parent directory to the path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_problematic_file(file_path):
    """
    Special handler for problematic files that can't be processed by normal means.
    Uses a simple line-by-line approach to extract data.
    """
    logger.info(f"Using special handler for problematic file: {file_path.name}")
    
    try:
        # For large files, process in chunks
        if file_path.stat().st_size > 50 * 1024 * 1024:  # If file is larger than 50MB
            logger.info(f"Large file detected ({file_path.stat().st_size / (1024 * 1024):.2f} MB). Processing in chunks...")
            data = []
            current_email = {"subject": "", "body": "", "label": "phishing" if "phish" in file_path.name.lower() else "unknown"}
            in_body = False
            
            # Process file in larger chunks with progress bar
            with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                # Get total size for progress bar
                total_size = file_path.stat().st_size
                chunk_size = 10 * 1024 * 1024  # 10MB chunks for better performance
                
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Reading {file_path.name}") as pbar:
                    # Read and process in chunks
                    buffer = ""
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                            
                        # Add the chunk to our buffer
                        buffer += chunk
                        
                        # Process complete lines from the buffer
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                if current_email["subject"] or current_email["body"]:
                                    data.append(current_email.copy())
                                    current_email = {"subject": "", "body": "", "label": "phishing" if "phish" in file_path.name.lower() else "unknown"}
                                in_body = False
                            elif line.startswith("Subject:"):
                                current_email["subject"] = line[8:].strip()
                                in_body = False
                            elif line.startswith("From:"):
                                current_email["from"] = line[5:].strip()
                                in_body = False
                            elif line.startswith("To:"):
                                current_email["receiver"] = line[3:].strip()
                                in_body = False
                            elif line.startswith("Date:"):
                                current_email["date"] = line[5:].strip()
                                in_body = False
                            else:
                                if not in_body:
                                    in_body = True
                                    current_email["body"] = line
                                else:
                                    current_email["body"] += " " + line
                        
                        pbar.update(len(chunk))
            
            # Process any remaining content in the buffer
            if buffer.strip():
                line = buffer.strip()
                if line.startswith("Subject:"):
                    current_email["subject"] = line[8:].strip()
                elif line.startswith("From:"):
                    current_email["from"] = line[5:].strip()
                elif line.startswith("To:"):
                    current_email["receiver"] = line[3:].strip()
                elif line.startswith("Date:"):
                    current_email["date"] = line[5:].strip()
                else:
                    if not in_body:
                        current_email["body"] = line
                    else:
                        current_email["body"] += " " + line
            
            # Add the last email if it exists
            if current_email["subject"] or current_email["body"]:
                data.append(current_email)
                
            # Create DataFrame more efficiently
            df = pd.DataFrame(data)
            logger.info(f"Successfully extracted {len(df)} emails from {file_path.name}")
            return df
            
        else:
            # For smaller files, use the original approach
            with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                content = f.read()
            
            # Split into lines
            lines = content.splitlines()
            
            # For Nazario.csv, we know it's phishing emails
            if file_path.name == 'Nazario.csv':
                # Create a simple DataFrame with subject, body and label
                data = []
                current_email = {"subject": "", "body": "", "label": "phishing"}
                in_body = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith("Subject:"):
                        # If we have a previous email, save it
                        if current_email["subject"] or current_email["body"]:
                            data.append(current_email.copy())
                            current_email = {"subject": "", "body": "", "label": "phishing"}
                        
                        current_email["subject"] = line[8:].strip()
                        in_body = False
                    elif line.startswith("From:"):
                        current_email["from"] = line[5:].strip()
                        in_body = False
                    elif line.startswith("To:"):
                        current_email["receiver"] = line[3:].strip()
                        in_body = False
                    elif line.startswith("Date:"):
                        current_email["date"] = line[5:].strip()
                        in_body = False
                    else:
                        # If not a header, it's part of the body
                        if not in_body:
                            in_body = True
                            current_email["body"] = line
                        else:
                            current_email["body"] += " " + line
                
                # Add the last email if it exists
                if current_email["subject"] or current_email["body"]:
                    data.append(current_email)
                    
                # Create DataFrame
                df = pd.DataFrame(data)
                logger.info(f"Successfully extracted {len(df)} emails from {file_path.name}")
                return df
                
            # For other problematic files, try a generic approach
            else:
                # Try to detect if it's a CSV by counting commas
                comma_count = sum(line.count(',') for line in lines[:20])
                if comma_count > 20:  # Likely a CSV
                    # Try to parse as CSV
                    data = []
                    for line in lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                data.append({
                                    "subject": parts[0].strip() if len(parts) > 0 else "",
                                    "body": parts[1].strip() if len(parts) > 1 else "",
                                    "label": "phishing" if "phish" in file_path.name.lower() else "unknown"
                                })
                    
                    df = pd.DataFrame(data)
                    logger.info(f"Successfully extracted {len(df)} records from {file_path.name}")
                    return df
                else:
                    # Not a CSV, try to extract emails
                    data = []
                    current_email = {"subject": "", "body": "", "label": "unknown"}
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if current_email["subject"] or current_email["body"]:
                                data.append(current_email.copy())
                                current_email = {"subject": "", "body": "", "label": "unknown"}
                        elif line.startswith("Subject:"):
                            current_email["subject"] = line[8:].strip()
                        elif line.startswith("From:"):
                            current_email["from"] = line[5:].strip()
                        elif line.startswith("To:"):
                            current_email["receiver"] = line[3:].strip()
                        else:
                            current_email["body"] += " " + line
                    
                    # Add the last email if it exists
                    if current_email["subject"] or current_email["body"]:
                        data.append(current_email)
                    
                    df = pd.DataFrame(data)
                    logger.info(f"Successfully extracted {len(df)} records from {file_path.name}")
                    return df
    
    except Exception as e:
        logger.error(f"Error processing problematic file {file_path.name}: {str(e)}", exc_info=True)
        return pd.DataFrame()

def main():
    """Process the phishing email dataset and save the processed data."""
    start_time = time.time()
    
    # Path to the raw data
    raw_data_path = Path("data/raw/phishing-email-dataset")
    
    # Output path for processed data
    processed_data_path = Path("data/processed")
    processed_data_path.mkdir(exist_ok=True, parents=True)
    
    # Log available CSV files
    csv_files = list(raw_data_path.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {raw_data_path}")
    for file in csv_files:
        logger.info(f"  - {file.name} ({file.stat().st_size / (1024 * 1024):.2f} MB)")
    
    # Process each file separately first
    dfs = {}  # Store DataFrames for later combination
    failed_files = []  # Track files that failed processing
    
    # List of known problematic files that need special handling
    problematic_files = ["Nazario.csv", "phishing_email.csv"]
    
    # Add progress bar for file processing
    for file in tqdm(csv_files, desc="Processing files", unit="file"):
        file_start_time = time.time()
        try:
            logger.info(f"Processing {file.name}...")
            
            # Use special handler for known problematic files
            if file.name in problematic_files:
                logger.info(f"Using special handler for {file.name}")
                df = process_problematic_file(file)
            else:
                # Set a timeout for processing each file (5 minutes)
                max_processing_time = 300  # seconds
                
                try:
                    df = load_messages([file])
                    processing_time = time.time() - file_start_time
                    
                    if processing_time > max_processing_time:
                        logger.warning(f"Processing {file.name} took too long ({processing_time:.2f}s). Skipping.")
                        failed_files.append(file.name)
                        continue
                except Exception as e:
                    logger.error(f"Error loading {file.name}: {str(e)}")
                    # Try with special handler as fallback
                    logger.info(f"Trying special handler for {file.name}")
                    df = process_problematic_file(file)
            
            if df.empty:
                logger.warning(f"No data loaded from {file.name} - skipping")
                failed_files.append(file.name)
                continue
                
            # Filter and select only relevant fields for model training
            # For phishing_email.csv, we want to preserve all columns
            if file.name == 'phishing_email.csv':
                logger.info("Preserving all columns from phishing_email.csv as requested")
                df_filtered = df  # Keep all columns
            else:
                df_filtered = filter_relevant_fields(df)
            
            # Save processed data
            output_file = processed_data_path / f"{file.stem}_processed.csv"
            df_filtered.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file} ({len(df_filtered)} rows)")
            
            # Store DataFrame for later combination
            dfs[file.name] = df_filtered
            
            # Log dataset statistics
            logger.info(f"Dataset stats for {file.stem}:")
            logger.info(f"  - Rows: {len(df_filtered)}")
            logger.info(f"  - Columns: {len(df_filtered.columns)}")
            logger.info(f"  - Column names: {list(df_filtered.columns)}")
            
            # Count phishing vs legitimate emails
            if 'label' in df_filtered.columns:
                phishing_count = len(df_filtered[df_filtered['label'] == 'phishing'])
                legit_count = len(df_filtered[df_filtered['label'] == 'legitimate'])
                logger.info(f"  - Phishing emails: {phishing_count}")
                logger.info(f"  - Legitimate emails: {legit_count}")
            
            # Show top senders
            if 'from' in df_filtered.columns:
                top_senders = df_filtered['from'].value_counts().head(3)
                logger.info(f"  - Top senders: {dict(top_senders)}")
            
            # Show top subjects
            if 'subject' in df_filtered.columns:
                top_subjects = df_filtered['subject'].value_counts().head(3)
                logger.info(f"  - Top subjects: {dict(top_subjects)}")
            
            # Log processing time
            processing_time = time.time() - file_start_time
            logger.info(f"  - Processing time: {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
            failed_files.append(file.name)
            continue
    
    # Log summary
    total_time = time.time() - start_time
    logger.info(f"\nProcessing complete in {total_time:.2f} seconds")
    logger.info(f"Successfully processed {len(dfs)} files")
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
    
    return 0

def filter_relevant_fields(df):
    """
    Filter the DataFrame to include only fields relevant for model training.
    Note: phishing_email.csv is handled specially to preserve all its columns.
    
    Args:
        df: DataFrame with email data
        
    Returns:
        DataFrame with only relevant fields (or all fields for phishing_email.csv)
    """
    # Check if DataFrame is empty
    if df.empty:
        return df
    
    # Check if this is phishing_email.csv data - if so, return all columns
    if 'source_file' in df.columns:
        is_phishing_email = df['source_file'].str.contains('phishing_email', case=False).any()
        if is_phishing_email:
            logger.info("Detected data from phishing_email.csv - preserving all columns")
            return df
    
    # List of relevant fields for phishing detection
    relevant_fields = [
        'from',          # Sender's email address (important for domain analysis)
        'reply_to',      # Reply-to address (often different in phishing emails)
        'subject',       # Email subject (contains urgency indicators, suspicious keywords)
        'body',          # Email body content (primary source of phishing indicators)
        'label'          # Target variable (phishing or legitimate)
    ]
    
    # Additional potentially useful fields if they exist
    optional_fields = [
        'urls',          # Extracted URLs from the email
        'links',         # Alternative name for URLs
        'num_links',     # Number of links in the email
        'has_attachment', # Whether email has attachments
        'date',          # Email date/time for temporal analysis
        'receiver',      # Recipients (may indicate targeting patterns)
        'to',            # Alternative name for recipient
    ]
    
    # Select relevant fields that exist in the DataFrame
    available_fields = [field for field in relevant_fields if field in df.columns]
    
    # Add optional fields if they exist
    for field in optional_fields:
        if field in df.columns:
            available_fields.append(field)
    
    # Return DataFrame with only available relevant fields
    filtered_df = df[available_fields].copy() if available_fields else df.copy()
    
    # Ensure the DataFrame has the required fields
    required_fields = ['subject', 'body', 'label']
    missing_fields = [field for field in required_fields if field not in filtered_df.columns]
    if missing_fields:
        logger.warning(f"Required fields missing: {missing_fields}")
    
    # Return the filtered DataFrame
    return filtered_df

if __name__ == "__main__":
    sys.exit(main()) 