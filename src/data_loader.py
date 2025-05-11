import os
import glob
import pandas as pd
import email
import chardet
import csv
from email import policy
from pathlib import Path
import logging
import re
from typing import List, Union, Optional
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_messages(paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load email messages from a list of file paths or directories, normalize headers,
    and return a cleaned DataFrame with standardized columns.
    
    Args:
        paths: List of file or directory paths containing .eml files or CSV files
        
    Returns:
        DataFrame with normalized columns including from, reply_to, subject, body, label
    """
    all_data = []
    
    for path in paths:
        path = Path(path)
        
        if path.is_dir():
            # Process directory - look for .eml and .csv files
            eml_files = list(path.glob('**/*.eml'))
            csv_files = list(path.glob('**/*.csv'))
            
            # Process .eml files if found
            if eml_files:
                logger.info(f"Processing {len(eml_files)} .eml files from {path}")
                for eml_path in eml_files:
                    try:
                        message_data = _process_eml_file(eml_path)
                        if message_data:
                            all_data.append(message_data)
                    except Exception as e:
                        logger.error(f"Error processing {eml_path}: {str(e)}")
            
            # Process .csv files if found
            if csv_files:
                logger.info(f"Processing {len(csv_files)} .csv files from {path}")
                for csv_path in csv_files:
                    try:
                        # Use different handlers based on filename
                        if csv_path.name == 'phishing_email.csv':
                            # Special handling for phishing_email.csv - preserve all columns
                            csv_data = _process_phishing_email_csv(csv_path)
                        elif _is_special_file(csv_path):
                            csv_data = _process_special_csv_file(csv_path)
                        else:
                            csv_data = _process_csv_file(csv_path)
                            
                        if not csv_data.empty:
                            all_data.append(csv_data)
                    except Exception as e:
                        logger.error(f"Error processing {csv_path}: {str(e)}")
        
        elif path.is_file():
            # Process a single file based on extension
            if path.suffix.lower() == '.eml':
                try:
                    message_data = _process_eml_file(path)
                    if message_data:
                        all_data.append(message_data)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
            
            elif path.suffix.lower() == '.csv':
                try:
                    # Use different handlers based on filename
                    if path.name == 'phishing_email.csv':
                        # Special handling for phishing_email.csv - preserve all columns
                        csv_data = _process_phishing_email_csv(path)
                    elif _is_special_file(path):
                        csv_data = _process_special_csv_file(path)
                    else:
                        csv_data = _process_csv_file(path)
                        
                    if not csv_data.empty:
                        all_data.append(csv_data)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
            else:
                logger.warning(f"Unsupported file type: {path}")
        
        else:
            logger.warning(f"Path does not exist: {path}")
    
    # Combine all data into a single DataFrame
    if not all_data:
        logger.warning("No valid data was found or loaded")
        return pd.DataFrame(columns=['from', 'reply_to', 'subject', 'body', 'label'])
    
    # If we have a mix of DataFrames and dicts, convert dicts to DataFrames
    processed_data = []
    for item in all_data:
        if isinstance(item, dict):
            processed_data.append(pd.DataFrame([item]))
        elif isinstance(item, pd.DataFrame):
            processed_data.append(item)
    
    # Combine all DataFrames
    df = pd.concat(processed_data, ignore_index=True)
    
    # Clean and standardize the DataFrame
    return _clean_dataframe(df)

def _is_special_file(file_path: Path) -> bool:
    """Check if the file needs special processing based on the filename."""
    special_files = ['Enron.csv', 'Ling.csv', 'Nazario.csv', 'Nigerian_Fraud.csv']
    return file_path.name in special_files

def _detect_encoding(file_path: Path) -> str:
    """Detect the encoding of a file using chardet."""
    # Read a sample of the file to detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
    
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'  # Default to utf-8 if detection fails
    
    # Handle common encoding issues
    if encoding.lower() in ['ascii', 'iso-8859-1'] and result['confidence'] < 0.9:
        # If low confidence in ASCII or ISO detection, use UTF-8 with error handling
        encoding = 'utf-8'
        
    # Handle confidence levels
    if result['confidence'] < 0.7:
        logger.warning(f"Low confidence ({result['confidence']:.2f}) in encoding detection for {file_path}. Using {encoding}.")
    
    logger.info(f"Detected encoding for {file_path.name}: {encoding} with confidence {result['confidence']:.2f}")
    return encoding

def _process_phishing_email_csv(file_path: Path) -> pd.DataFrame:
    """
    Special handler for phishing_email.csv to preserve all its columns.
    This file contains important features for phishing detection.
    """
    try:
        # Try multiple encodings, starting with latin-1 which can handle most byte sequences
        encodings_to_try = ['latin-1', 'utf-8', 'cp1252']
        
        df = None
        last_error = None
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read phishing_email.csv with {encoding} encoding")
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False, 
                               on_bad_lines='skip', quoting=3)  # quoting=3 disables quoting
                if df is not None and not df.empty:
                    logger.info(f"Successfully read phishing_email.csv with {encoding} encoding")
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to read phishing_email.csv with {encoding} encoding: {str(e)}")
                continue
        
        if df is None or df.empty:
            logger.warning(f"Standard approaches failed for phishing_email.csv")
            logger.info("Trying fallback method for phishing_email.csv")
            
            # Fallback: Try to read the file in chunks
            try:
                chunks = []
                chunk_size = 1000  # Process 1000 rows at a time
                
                for chunk in pd.read_csv(file_path, encoding='latin-1', chunksize=chunk_size, 
                                       on_bad_lines='skip', quoting=3, low_memory=False):
                    chunks.append(chunk)
                
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    logger.info(f"Successfully loaded phishing_email.csv in chunks: {len(df)} rows")
                else:
                    logger.error(f"Failed to extract data from {file_path} using chunk method")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Chunk method failed for phishing_email.csv: {str(e)}")
                
                # Final fallback: Use CSV module directly
                try:
                    logger.info("Trying CSV module directly for phishing_email.csv")
                    return _process_csv_file(file_path)
                except Exception as e:
                    logger.error(f"All methods failed for phishing_email.csv: {str(e)}")
                    return pd.DataFrame()
            
        # Preserve all columns, but ensure we have the essential ones
        if 'text' in df.columns and 'body' not in df.columns:
            df['body'] = df['text']
        
        if 'class' in df.columns and 'label' not in df.columns:
            df['label'] = df['class']
            
        logger.info(f"Successfully loaded phishing_email.csv with {len(df.columns)} columns: {df.columns.tolist()}")
        
        # Clean up any rows with NaN values in critical columns
        if 'body' in df.columns:
            df = df.dropna(subset=['body'])
            
        if 'label' in df.columns:
            # Map any numeric labels to strings
            if df['label'].dtype in ['int64', 'float64']:
                df['label'] = df['label'].map({1: 'phishing', 0: 'legitimate'})
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to process phishing_email.csv: {str(e)}")
        return pd.DataFrame()

def _process_eml_file(file_path: Path) -> dict:
    """Extract relevant fields from an .eml file."""
    try:
        # Detect encoding
        encoding = _detect_encoding(file_path)
        
        # Parse the email
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            msg = email.message_from_file(f, policy=policy.default)
        
        # Extract data
        from_addr = msg.get('From', '')
        reply_to = msg.get('Reply-To', '')
        subject = msg.get('Subject', '')
        
        # Get the message body
        body = ''
        if msg.is_multipart():
            # Handle multipart messages by finding the first text/plain part
            for part in msg.iter_parts():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    break
        else:
            # For simple messages
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                body = msg.get_content()
            elif content_type == 'text/html':
                # Basic HTML to text conversion (a more sophisticated approach would use a library)
                body = msg.get_content()
                # Strip HTML tags (simple approach)
                import re
                body = re.sub(r'<[^>]+>', ' ', body)
        
        # Try to determine if it's phishing or legitimate based on the filename or path
        # This is a simple heuristic and should be adjusted based on your dataset
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        if 'phish' in file_name or 'spam' in file_name or 'phish' in path_str:
            label = 'phishing'
        elif 'legit' in file_name or 'ham' in file_name or 'legit' in path_str:
            label = 'legitimate'
        else:
            label = None  # Unknown
            
        return {
            'from': from_addr,
            'reply_to': reply_to,
            'subject': subject,
            'body': body, 
            'label': label,
            'source_file': str(file_path)
        }
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        raise

def _process_special_csv_file(file_path: Path) -> pd.DataFrame:
    """Process known CSV files with special handling for each format."""
    try:
        # Detect encoding
        encoding = _detect_encoding(file_path)
        file_name = file_path.name
        
        # Try with latin-1 encoding first as it can handle most byte sequences
        try:
            if file_name == 'Enron.csv':
                # Enron dataset has 3 columns: subject, body, label
                df = pd.read_csv(file_path, encoding='latin-1', names=['subject', 'body', 'label'], 
                                on_bad_lines='skip', quoting=3)  # quoting=3 disables quoting
                # Map labels: 0 = legitimate, 1 = phishing
                df['label'] = df['label'].map({0: 'legitimate', 1: 'phishing'})
                
            elif file_name == 'Ling.csv':
                # Ling dataset has 3 columns: subject, body, label
                df = pd.read_csv(file_path, encoding='latin-1', names=['subject', 'body', 'label'], 
                                on_bad_lines='skip', quoting=3)
                # Map labels: 0 = legitimate, 1 = phishing
                df['label'] = df['label'].map({0: 'legitimate', 1: 'phishing'})
                
            elif file_name == 'Nazario.csv':
                # Nazario dataset has 7 columns: sender, receiver, date, subject, body, urls, label
                df = pd.read_csv(file_path, encoding='latin-1', 
                                names=['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label'],
                                on_bad_lines='skip', quoting=3)
                # Rename sender to from
                df = df.rename(columns={'sender': 'from'})
                # Map labels: 1 = phishing
                df['label'] = df['label'].map({1: 'phishing'})
                
            elif file_name == 'Nigerian_Fraud.csv':
                # Nigerian_Fraud dataset has 7 columns: sender, receiver, date, subject, body, urls, label
                df = pd.read_csv(file_path, encoding='latin-1', 
                                names=['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label'],
                                on_bad_lines='skip', quoting=3)
                # Rename sender to from
                df = df.rename(columns={'sender': 'from'})
                # Map labels: 1 = phishing
                df['label'] = df['label'].map({1: 'phishing'})
                
            else:
                # Fallback to generic processing
                df = _process_csv_file(file_path)
                
            return df
        except Exception as e:
            logger.warning(f"Failed with latin-1 encoding for {file_name}: {str(e)}")
            # Try with the detected encoding as fallback
            return _process_csv_file(file_path)
        
    except Exception as e:
        logger.error(f"Failed to process special CSV {file_path}: {str(e)}")
        # Try generic processing as fallback
        return _process_csv_file(file_path)

def _process_binary_csv(file_path: Path) -> pd.DataFrame:
    """
    Process CSV files that might be binary or have unusual formats.
    This is a last resort for files that can't be processed by other methods.
    """
    try:
        logger.info(f"Attempting binary processing for {file_path.name}")
        
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decode using various methods
        text = None
        for encoding in ['latin-1', 'utf-8', 'cp1252', 'ascii']:
            try:
                text = content.decode(encoding, errors='replace')
                break
            except Exception:
                continue
                
        if not text:
            logger.error(f"Could not decode {file_path.name} with any encoding")
            return pd.DataFrame()
        
        # Split into lines and try to parse as CSV
        lines = text.splitlines()
        if not lines:
            logger.error(f"No lines found in {file_path.name}")
            return pd.DataFrame()
            
        # Try to determine the delimiter
        possible_delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        for delim in possible_delimiters:
            delimiter_counts[delim] = sum(line.count(delim) for line in lines[:20])
        
        # Use the delimiter that appears most consistently
        delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        # Parse the CSV data
        rows = []
        for line in lines:
            if line.strip():  # Skip empty lines
                rows.append(line.split(delimiter))
        
        if not rows:
            logger.error(f"No valid rows found in {file_path.name}")
            return pd.DataFrame()
            
        # Create a DataFrame
        headers = rows[0] if rows else []
        
        # Clean up headers
        clean_headers = []
        for i, header in enumerate(headers):
            header = header.strip()
            if not header:
                header = f"column_{i}"
            clean_headers.append(header)
        
        # Process data rows
        data = []
        for row in rows[1:]:
            # Ensure row has same length as headers
            if len(row) < len(clean_headers):
                row = row + [''] * (len(clean_headers) - len(row))
            elif len(row) > len(clean_headers):
                row = row[:len(clean_headers)]
            
            # Clean up values
            clean_row = [val.strip() for val in row]
            data.append(clean_row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=clean_headers)
        
        # Try to identify key columns
        for col in df.columns:
            col_lower = col.lower()
            if 'subject' in col_lower or 'title' in col_lower:
                df['subject'] = df[col]
            elif 'body' in col_lower or 'content' in col_lower or 'text' in col_lower or 'message' in col_lower:
                df['body'] = df[col]
            elif 'label' in col_lower or 'class' in col_lower or 'type' in col_lower or 'spam' in col_lower:
                df['label'] = df[col]
        
        logger.info(f"Successfully processed binary file {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Binary processing failed for {file_path.name}: {str(e)}")
        return pd.DataFrame()

def _process_csv_file(file_path: Path) -> pd.DataFrame:
    """Load and preprocess data from a CSV file."""
    try:
        # Detect encoding
        encoding = _detect_encoding(file_path)
        
        # Try multiple approaches with error handling for encoding issues
        approaches = [
            # First try with auto-detection and errors='replace'
            lambda: pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', 
                               on_bad_lines='skip', quoting=3, errors='replace'),
            
            # Try with specific encoding and errors='replace'
            lambda: pd.read_csv(file_path, encoding=encoding, sep=',', 
                               on_bad_lines='skip', quoting=3, errors='replace'),
            
            # Try with UTF-8 and errors='replace'
            lambda: pd.read_csv(file_path, encoding='utf-8', sep=',', 
                               on_bad_lines='skip', quoting=3, errors='replace'),
            
            # Try with latin-1 (should handle most byte sequences)
            lambda: pd.read_csv(file_path, encoding='latin-1', sep=',', 
                               on_bad_lines='skip', quoting=3)
        ]
        
        df = None
        last_error = None
        
        for approach_func in approaches:
            try:
                df = approach_func()
                if df is not None and not df.empty and len(df.columns) > 1:
                    break
            except Exception as e:
                last_error = e
                continue
        
        if df is None or df.empty or len(df.columns) <= 1:
            logger.warning(f"Standard approaches failed for {file_path}: {last_error}")
            logger.info(f"Trying fallback method with Python's csv module for {file_path.name}")
            
            # Fallback: Use Python's csv module directly for problematic files
            try:
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
                    
                    for row in reader:
                        if row:  # Skip empty rows
                            rows.append(row)
                
                if rows:
                    # Use the first row as header
                    headers = rows[0] if len(rows) > 0 else []
                    
                    # If headers are empty strings or numbers, generate column names
                    for i in range(len(headers)):
                        if not headers[i] or headers[i].isdigit():
                            headers[i] = f"column_{i}"
                    
                    # Create DataFrame from the data
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    # Ensure all rows have the same length as the header
                    uniform_rows = []
                    for row in data_rows:
                        # Pad or truncate rows to match header length
                        if len(row) < len(headers):
                            uniform_rows.append(row + [''] * (len(headers) - len(row)))
                        else:
                            uniform_rows.append(row[:len(headers)])
                    
                    df = pd.DataFrame(uniform_rows, columns=headers)
                    logger.info(f"Successfully loaded {len(df)} rows using fallback method")
                    
                    # Try to determine which columns might be the ones we need
                    # Look for columns that might contain subject, body, and label
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'subject' in col_lower or 'title' in col_lower:
                            df['subject'] = df[col]
                        elif 'body' in col_lower or 'content' in col_lower or 'text' in col_lower or 'message' in col_lower:
                            df['body'] = df[col]
                        elif 'label' in col_lower or 'class' in col_lower or 'type' in col_lower or 'spam' in col_lower:
                            df['label'] = df[col]
                    
                    return df
                else:
                    logger.warning(f"No rows found in {file_path} using fallback method")
                    # Try binary processing as a last resort
                    return _process_binary_csv(file_path)
                    
            except Exception as e:
                logger.error(f"Fallback method failed for {file_path}: {str(e)}")
                # Try binary processing as a last resort
                return _process_binary_csv(file_path)
        
        return df
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {str(e)}")
        # Try binary processing as a last resort
        return _process_binary_csv(file_path)

def _normalize_column_name(col_name: str) -> str:
    """Convert column names to lowercase snake_case."""
    # Replace spaces and special characters with underscores
    import re
    normalized = re.sub(r'[^a-zA-Z0-9]', '_', str(col_name).lower())
    # Replace multiple underscores with a single one
    normalized = re.sub(r'_+', '_', normalized)
    # Remove leading and trailing underscores
    normalized = normalized.strip('_')
    return normalized

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the DataFrame."""
    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame(columns=['from', 'reply_to', 'subject', 'body', 'label'])
    
    # Preserve phishing_email.csv if it's the source
    if 'source_file' in df.columns:
        is_phishing_email_csv = df['source_file'].str.contains('phishing_email.csv', case=False).any()
        if is_phishing_email_csv:
            logger.info("Preserving all columns from phishing_email.csv")
            # Only perform essential cleaning
            # Ensure label is properly formatted
            if 'label' in df.columns:
                df['label'] = df['label'].astype(str).str.lower()
                # Map numeric labels
                if df['label'].str.contains(r'^[01]\.?0*$', regex=True).all():
                    df['label'] = df['label'].map(
                        lambda x: 'phishing' if x.startswith('1') else 'legitimate'
                    )
            return df
    
    # Check for completely numeric column names and replace them
    df.columns = [f'col_{i}' if isinstance(c, (int, float)) or str(c).isdigit() else c 
                 for i, c in enumerate(df.columns)]
    
    # Normalize column names to lowercase snake_case
    df.columns = [_normalize_column_name(col) for col in df.columns]
    
    # Map common column variations to standard names
    column_mapping = {
        'sender': 'from',
        'from_address': 'from',
        'from_addr': 'from',
        'sender_email': 'from',
        'email_from': 'from',
        
        'reply_to_address': 'reply_to',
        'replyto': 'reply_to',
        
        'email_subject': 'subject',
        'mail_subject': 'subject',
        
        'content': 'body',
        'email_body': 'body',
        'message': 'body',
        'text': 'body',
        'email_content': 'body',
        
        'is_phishing': 'label',
        'is_spam': 'label',
        'class': 'label',
        'category': 'label',
        'type': 'label'
    }
    
    # Apply the mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
        elif old_col in df.columns and new_col in df.columns:
            # Keep the existing column, fill gaps if needed
            df[new_col] = df[new_col].fillna(df[old_col])
    
    # Ensure all required columns exist
    required_cols = ['from', 'subject', 'body', 'label']
    for col in required_cols:
        if col not in df.columns:
            if col == 'from' and 'sender' in df.columns:
                df[col] = df['sender']
            else:
                df[col] = None
    
    # Add reply_to if it doesn't exist
    if 'reply_to' not in df.columns:
        df['reply_to'] = None
    
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
            logger.warning(f"Found unknown labels: {unknown_labels}. These will be set to None.")
            df.loc[~mask, 'label'] = None
    
    # Drop rows with missing labels
    df = df.dropna(subset=['label'])
    
    # Drop rows with empty bodies - unless there's a non-empty subject
    body_mask = df['body'].isna() | (df['body'] == '')
    subject_mask = df['subject'].isna() | (df['subject'] == '')
    drop_mask = body_mask & subject_mask
    if drop_mask.any():
        logger.info(f"Dropping {drop_mask.sum()} rows with empty content (both body and subject)")
        df = df[~drop_mask].copy()
    
    # Handle NaN values
    df = df.fillna('')
    
    # Clean text fields - remove excessive whitespace
    for col in ['subject', 'body']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Drop duplicate rows based on combination of from, subject, and body
    df = df.drop_duplicates(subset=['from', 'subject', 'body'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_paths = ["data/raw"]
    messages_df = load_messages(sample_paths)
    print(f"Loaded {len(messages_df)} messages")
    print(messages_df.columns)
    print(messages_df.head()) 