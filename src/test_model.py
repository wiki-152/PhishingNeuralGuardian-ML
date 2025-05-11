#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the trained phishing detection model on sample emails.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import argparse
import email
from email import policy
from pathlib import Path
from predict_phishing import predict_email, load_model_and_vectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_email_content(email_path):
    """
    Extract content from an email file (.eml format).
    
    Args:
        email_path: Path to the email file
        
    Returns:
        Dictionary with email fields
    """
    try:
        with open(email_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        # Extract subject
        subject = msg['subject'] or ""
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                # Get text content
                if content_type == 'text/plain' or content_type == 'text/html':
                    try:
                        body += part.get_content()
                    except:
                        body += str(part.get_payload(decode=True))
        else:
            # Not multipart - get content directly
            body = msg.get_content() or str(msg.get_payload(decode=True))
        
        # Extract other fields
        from_field = msg['from'] or ""
        reply_to = msg['reply-to'] or ""
        
        return {
            'subject': subject,
            'body': body,
            'from': from_field,
            'reply_to': reply_to
        }
    except Exception as e:
        logger.error(f"Error extracting email content from .eml file: {e}")
        return {
            'subject': "",
            'body': "",
            'from': "",
            'reply_to': ""
        }

def extract_text_email_content(email_path):
    """
    Extract content from a text file, expecting a certain format.
    Assumes the first line is the subject, second line is blank,
    and the rest is the body.
    
    Args:
        email_path: Path to the text file
        
    Returns:
        Dictionary with email fields
    """
    try:
        with open(email_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract subject (first line, assuming it might be "Subject: Something")
        subject_line = lines[0].strip()
        if subject_line.lower().startswith('subject:'):
            subject = subject_line[len('subject:'):].strip()
        else:
            subject = subject_line
        
        # Extract body (all lines after the first or second)
        if len(lines) > 1 and lines[1].strip() == "":
            body_text = ''.join(lines[2:])
        else:
            body_text = ''.join(lines[1:])
        
        # Look for From/Reply-To fields
        from_field = ""
        reply_to = ""
        
        for line in lines:
            if line.lower().startswith('from:'):
                from_field = line[len('from:'):].strip()
            elif line.lower().startswith('reply-to:'):
                reply_to = line[len('reply-to:'):].strip()
        
        return {
            'subject': subject,
            'body': body_text,
            'from': from_field,
            'reply_to': reply_to
        }
    except Exception as e:
        logger.error(f"Error extracting content from text file: {e}")
        return {
            'subject': "",
            'body': "",
            'from': "",
            'reply_to': ""
        }

def extract_csv_email(csv_path, index=0):
    """
    Extract email content from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        index: Index of the email to extract (if multiple)
        
    Returns:
        Dictionary with email fields
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if the CSV has multiple rows
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        
        # Use the specified index or the first row
        if index >= len(df):
            logger.warning(f"Index {index} out of range, using first email")
            index = 0
        
        row = df.iloc[index]
        
        # Required fields
        subject = row.get('subject', '')
        body = row.get('body', '')
        
        # Optional fields
        from_field = row.get('from', '')
        reply_to = row.get('reply_to', '')
        
        return {
            'subject': subject,
            'body': body,
            'from': from_field,
            'reply_to': reply_to
        }
    except Exception as e:
        logger.error(f"Error extracting email from CSV: {e}")
        return {
            'subject': "",
            'body': "",
            'from': "",
            'reply_to': ""
        }

def get_email_content(file_path, index=0):
    """
    Extract email content based on the file extension.
    
    Args:
        file_path: Path to the email file
        index: Index to use for CSV files with multiple emails
    
    Returns:
        Dictionary with email fields
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.eml':
        return extract_email_content(file_path)
    elif file_path.suffix.lower() == '.txt':
        return extract_text_email_content(file_path)
    elif file_path.suffix.lower() == '.csv':
        return extract_csv_email(file_path, index)
    else:
        logger.warning(f"Unsupported file format: {file_path.suffix}")
        # Try text extraction as a fallback
        return extract_text_email_content(file_path)

def test_email_file(model, vectorizer, email_path, scaler=None, index=0, threshold=None):
    """
    Test the model on a single email file.
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        email_path: Path to the email file
        scaler: Feature scaler (optional)
        index: Index for CSV files with multiple emails
        threshold: Custom threshold for classification (optional)
        
    Returns:
        prediction: 1 for phishing, 0 for legitimate
        probability: Probability of being phishing
    """
    logger.info(f"Testing email: {email_path}")
    
    # Extract email content based on file type
    email_data = get_email_content(email_path, index)
    
    # Print email details
    print("\nEmail Details:")
    print(f"Subject: {email_data['subject']}")
    print(f"From: {email_data['from']}")
    print(f"Reply-To: {email_data['reply_to']}")
    print(f"Body length: {len(email_data['body'])} characters")
    
    # Make prediction
    result = predict_email(model, vectorizer, email_data, scaler)
    
    if result is None:
        logger.error("Failed to get prediction")
        return None, 0.0
    
    prediction = result['predictions'][0]
    probability = result['probabilities'][0]
    
    # Determine result label based on threshold
    model_threshold = getattr(model, 'threshold_', 0.5)
    if threshold is not None:
        model_threshold = threshold
    
    # Override prediction based on custom threshold if provided
    if threshold is not None:
        prediction = 1 if probability >= threshold else 0
        
    result_label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    
    # Print result
    print(f"\nPrediction: {result_label}")
    print(f"Confidence: {probability:.4f} (threshold: {model_threshold:.4f})")
    
    return prediction, probability

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test phishing detection model on email files")
    
    parser.add_argument(
        "email_path",
        type=str,
        help="Path to email file (.eml, .txt, .csv) or directory containing email files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="../models/mlp_sklearn.pkl",
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="../models/tfidf.pkl",
        help="Path to the trained vectorizer"
    )
    
    parser.add_argument(
        "--scaler",
        type=str,
        default="../models/scaler.pkl",
        help="Path to the feature scaler"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Custom classification threshold (overrides model's threshold)"
    )
    
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the email to test in a CSV file (if it contains multiple emails)"
    )
    
    args = parser.parse_args()
    
    # Load model, vectorizer, and scaler
    model, vectorizer, scaler = load_model_and_vectorizer(
        args.model, args.vectorizer, args.scaler
    )
    
    # Override threshold if specified
    threshold = None
    if args.threshold is not None:
        logger.info(f"Using custom threshold: {args.threshold}")
        threshold = args.threshold
    
    # Check if path is a file or directory
    path = Path(args.email_path)
    if path.is_file():
        # Test single email file
        test_email_file(model, vectorizer, path, scaler, args.index, threshold)
    elif path.is_dir():
        # Find all supported files
        email_files = []
        for ext in ['.eml', '.txt', '.csv']:
            email_files.extend(list(path.glob(f"*{ext}")))
        
        if not email_files:
            logger.error(f"No supported email files found in directory: {path}")
            logger.info("Supported formats: .eml, .txt, .csv")
            return
        
        print(f"Testing {len(email_files)} email files...")
        
        results = []
        for email_file in email_files:
            try:
                prediction, probability = test_email_file(model, vectorizer, email_file, scaler, 
                                                         threshold=threshold)
                if prediction is not None:
                    results.append({
                        'file': email_file.name,
                        'prediction': prediction,
                        'probability': probability,
                        'label': "phishing" if prediction == 1 else "legitimate"
                    })
            except Exception as e:
                logger.error(f"Error processing {email_file}: {e}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total emails processed: {len(results)}")
        if results:
            phishing_count = sum(1 for r in results if r['prediction'] == 1)
            print(f"Phishing emails: {phishing_count}")
            print(f"Legitimate emails: {len(results) - phishing_count}")

if __name__ == "__main__":
    main() 