import os
import pandas as pd
import chardet
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        # Read a reasonable sample of the file to determine encoding
        raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))
    
    result = chardet.detect(raw_data)
    logger.info(f"Detected encoding for {file_path.name}: {result}")
    return result['encoding']

def examine_csv(file_path, rows=5):
    """Examine the structure of a CSV file."""
    file_path = Path(file_path)
    
    try:
        # Detect encoding
        encoding = detect_encoding(file_path)
        
        # Try to read with different separators
        for sep in [',', ';', '\t', '|']:
            try:
                logger.info(f"Trying separator: '{sep}'")
                # Try with header
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=rows, on_bad_lines='warn')
                if len(df.columns) > 1:
                    logger.info(f"Success with separator: '{sep}', found {len(df.columns)} columns")
                    logger.info(f"Columns: {df.columns.tolist()}")
                    logger.info(f"First {rows} rows:")
                    print(df.head(rows))
                    return
                
                # Try without header
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=rows, header=None, on_bad_lines='warn')
                if len(df.columns) > 1:
                    logger.info(f"Success with separator: '{sep}' (no header), found {len(df.columns)} columns")
                    logger.info(f"First {rows} rows:")
                    print(df.head(rows))
                    return
            except Exception as e:
                logger.warning(f"Error with separator '{sep}': {str(e)}")
        
        # If still not successful, try a more flexible approach
        logger.info("Trying with Python engine and automatic delimiter detection")
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', nrows=rows)
            logger.info(f"Success! Found {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"First {rows} rows:")
            print(df.head(rows))
        except Exception as e:
            logger.error(f"Failed with Python engine: {str(e)}")
            
            # Last resort: just read a few lines directly
            logger.info("Reading raw lines:")
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= rows:
                        break
                    print(f"Line {i+1}: {line.strip()}")
    
    except Exception as e:
        logger.error(f"Failed to examine {file_path}: {str(e)}")

if __name__ == "__main__":
    data_dir = Path("data/raw/phishing-email-dataset")
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        logger.info(f"Examining {file.name} ({file.stat().st_size / (1024 * 1024):.2f} MB)")
        examine_csv(file)
        print("\n" + "="*80 + "\n") 