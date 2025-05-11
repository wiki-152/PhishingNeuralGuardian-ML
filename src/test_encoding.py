#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify encoding fixes by loading problematic files.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Add the parent directory to the path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_messages, _detect_encoding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_problematic_files():
    """Test loading problematic files that had encoding issues."""
    raw_data_path = Path("data/raw/phishing-email-dataset")
    
    # List of files that had encoding issues
    problematic_files = [
        "Enron.csv",
        "Ling.csv", 
        "Nazario.csv",
        "Nigerian_Fraud.csv",
        "phishing_email.csv"
    ]
    
    results = {}
    
    for filename in problematic_files:
        file_path = raw_data_path / filename
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Testing encoding detection for {filename}")
        encoding = _detect_encoding(file_path)
        
        logger.info(f"Testing loading {filename}")
        try:
            df = load_messages([file_path])
            success = not df.empty
            row_count = len(df) if success else 0
            results[filename] = {
                "success": success,
                "rows": row_count,
                "encoding": encoding
            }
            logger.info(f"Result for {filename}: {'✓ Success' if success else '✗ Failed'} - {row_count} rows")
            
            if success and row_count > 0:
                # Display sample data
                logger.info(f"Sample data from {filename}:")
                if 'subject' in df.columns and 'body' in df.columns:
                    for i, row in df.head(2).iterrows():
                        subject = row.get('subject', '')[:50]
                        body = row.get('body', '')[:50]
                        logger.info(f"  Row {i}: subject='{subject}...', body='{body}...'")
                else:
                    logger.info(f"  Columns: {df.columns.tolist()}")
                    logger.info(f"  First row: {df.iloc[0].to_dict()}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}", exc_info=True)
            results[filename] = {
                "success": False,
                "error": str(e),
                "encoding": encoding
            }
    
    # Summary
    logger.info("\n===== SUMMARY =====")
    for filename, result in results.items():
        status = "✓ Success" if result.get("success", False) else "✗ Failed"
        rows = result.get("rows", 0)
        encoding = result.get("encoding", "unknown")
        logger.info(f"{filename}: {status} - {rows} rows - {encoding} encoding")
    
    return results

if __name__ == "__main__":
    test_problematic_files() 