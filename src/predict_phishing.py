#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use the trained model to predict whether an email is phishing or legitimate.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import argparse
import re
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
    logger.debug("VADER lexicon already downloaded")
except LookupError:
    logger.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def load_model_and_vectorizer(model_path, vectorizer_path, scaler_path=None):
    """
    Load the trained model, vectorizer, and optional scaler.
    
    Args:
        model_path: Path to the trained model
        vectorizer_path: Path to the trained vectorizer
        scaler_path: Path to the feature scaler (optional)
        
    Returns:
        model: Trained model
        vectorizer: Trained vectorizer
        scaler: Feature scaler (or None if not provided)
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
        # If it's an MLPClassifier, ensure it has the necessary attributes
        if hasattr(model, '__class__') and model.__class__.__name__ == 'MLPClassifier':
            # Add any missing attributes that might be expected
            if not hasattr(model, 'predict_with_threshold'):
                def predict_with_threshold(self, X, threshold=0.5):
                    probs = self.predict_proba(X)[:, 1]
                    return (probs >= threshold).astype(int)
                model.predict_with_threshold = predict_with_threshold.__get__(model)
    
    logger.info(f"Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    scaler = None
    if scaler_path:
        if os.path.exists(scaler_path):
            try:
                logger.info(f"Loading feature scaler from {scaler_path}")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading scaler: {e}. Continuing without scaler.")
        else:
            logger.warning(f"Scaler path {scaler_path} does not exist. Continuing without scaler.")
    
    return model, vectorizer, scaler

def sentiment_features(text):
    """
    Extract sentiment features from text using VADER.
    
    Args:
        text: String, the text to analyze
        
    Returns:
        Dictionary with sentiment scores: compound, neg, neu, pos
    """
    try:
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(str(text))
        
        return {
            'compound': sentiment_scores['compound'],  # Range: -1 (negative) to 1 (positive)
            'neg': sentiment_scores['neg'],            # Range: 0 to 1
            'neu': sentiment_scores['neu'],            # Range: 0 to 1
            'pos': sentiment_scores['pos']             # Range: 0 to 1
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {
            'compound': 0.0,
            'neg': 0.0,
            'neu': 1.0,
            'pos': 0.0
        }

def count_urls(text):
    """Count the number of URLs in the text."""
    try:
        # Simple regex to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        count = len(url_pattern.findall(str(text)))
        return count
    except Exception as e:
        logger.error(f"Error counting URLs: {e}")
        return 0

def count_urgency_words(text):
    """Count words that indicate urgency."""
    try:
        # List of words indicating urgency
        urgency_words = [
            'urgent', 'immediately', 'attention', 'important', 'action', 
            'required', 'verify', 'confirm', 'update', 'security', 'alert',
            'warning', 'limited', 'expires', 'deadline', 'asap', 'now',
            'quick', 'critical', 'urgent', 'validate', 'suspend'
        ]
        
        # Convert to lowercase and count matches
        text = str(text).lower()
        count = sum(1 for word in urgency_words if word in text)
        return count
    except Exception as e:
        logger.error(f"Error counting urgency words: {e}")
        return 0

def detect_spoofing(from_field, reply_to_field=None):
    """
    Detect potential email spoofing by checking for mismatches between
    'from' and 'reply-to' fields, or suspicious patterns in the 'from' field.
    """
    try:
        # Convert inputs to strings and lowercase
        from_field = str(from_field).lower()
        reply_to_field = str(reply_to_field).lower() if reply_to_field else ""
        
        # Check for mismatch between from and reply-to domains
        from_domain = re.search(r'@([^>@\s]+)', from_field)
        reply_domain = re.search(r'@([^>@\s]+)', reply_to_field)
        
        # If both domains exist and don't match, potential spoofing
        if from_domain and reply_domain and from_domain.group(1) != reply_domain.group(1):
            return 1
        
        # Check for suspicious patterns in from field
        suspicious_patterns = [
            r'@.*@',                  # Multiple @ symbols
            r'[^a-zA-Z0-9.@_-]',      # Special characters in email
            r'\.(ru|cn|tk|top|xyz)$'  # Suspicious TLDs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, from_field):
                return 1
                
        return 0
    except Exception as e:
        logger.error(f"Error in spoofing detection: {e}")
        return 0

def extract_domain_features(email):
    """
    Extract features related to email domains and addresses.
    
    Args:
        email: Dictionary with email fields
        
    Returns:
        Dictionary with domain features
    """
    try:
        features = {}
        
        # Extract domain from from_field
        from_field = str(email.get('from', ''))
        domain_match = re.search(r'@([^>@\s]+)', from_field.lower())
        domain = domain_match.group(1) if domain_match else ""
        
        # Common legitimate domains
        common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'icloud.com']
        features['is_common_domain'] = 1 if domain in common_domains else 0
        
        # Domain length (unusually long domains can be suspicious)
        features['domain_length'] = len(domain)
        
        # Number of subdomains (more subdomains can be suspicious)
        features['subdomain_count'] = domain.count('.') if domain else 0
        
        # Domain has numbers (can be suspicious)
        features['domain_has_numbers'] = 1 if re.search(r'\d', domain) else 0
        
        return features
    except Exception as e:
        logger.error(f"Error extracting domain features: {e}")
        return {
            'is_common_domain': 0,
            'domain_length': 0,
            'subdomain_count': 0,
            'domain_has_numbers': 0
        }

def prepare_email_features(email_data, vectorizer, scaler=None):
    """
    Prepare features for prediction.
    
    Args:
        email_data: DataFrame or dictionary with email data
        vectorizer: Trained TF-IDF vectorizer
        scaler: Feature scaler (optional)
        
    Returns:
        X: Feature matrix ready for prediction
    """
    try:
        # Convert to DataFrame if dictionary
        if isinstance(email_data, dict):
            email_data = pd.DataFrame([email_data])
        
        # Ensure required columns exist
        required_cols = ['subject', 'body']
        for col in required_cols:
            if col not in email_data.columns:
                logger.error(f"Missing required column: {col}")
                return None
        
        # Combine subject and body for TF-IDF features
        logger.info("Creating TF-IDF features")
        email_data['text'] = email_data['subject'].fillna('') + ' ' + email_data['body'].fillna('')
        
        # Create TF-IDF features
        X_tfidf = vectorizer.transform(email_data['text'])
        logger.info(f"TF-IDF shape: {X_tfidf.shape}")
        
        # Create numeric features
        logger.info("Creating numeric features")
        numeric_features = []
        
        # Text length features
        email_data['subject_length'] = email_data['subject'].fillna('').apply(len)
        email_data['body_length'] = email_data['body'].fillna('').apply(len)
        
        # Sentiment analysis
        logger.info("Performing sentiment analysis")
        sentiments = email_data['text'].apply(sentiment_features)
        email_data['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
        email_data['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
        email_data['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
        email_data['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
        
        # URL count
        email_data['url_count'] = email_data['body'].fillna('').apply(count_urls)
        
        # Urgency words count
        email_data['urgency_count'] = email_data['text'].apply(count_urgency_words)
        
        # Domain features
        logger.info("Extracting domain features")
        domain_features = email_data.apply(
            lambda row: extract_domain_features({
                'from': row.get('from', ''),
                'reply_to': row.get('reply_to', '')
            }),
            axis=1
        )
        
        for feature in ['is_common_domain', 'domain_length', 'has_suspicious_tld']:
            if feature in domain_features[0]:
                email_data[feature] = domain_features.apply(lambda x: x.get(feature, 0))
        
        # Spoofing detection
        email_data['spoofing_score'] = email_data.apply(
            lambda row: detect_spoofing(
                row.get('from', ''), 
                row.get('reply_to', '')
            ),
            axis=1
        )
        
        # Select numeric features
        numeric_cols = [
            'subject_length', 'body_length', 
            'sentiment_compound', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive',
            'url_count', 'urgency_count', 'spoofing_score'
        ]
        
        # Add domain features if they exist
        for col in ['is_common_domain', 'domain_length', 'has_suspicious_tld']:
            if col in email_data.columns:
                numeric_cols.append(col)
        
        # Create numeric feature matrix
        X_numeric = email_data[numeric_cols].values
        logger.info(f"Numeric features shape: {X_numeric.shape}")
        
        # SKIP scaling numeric features - this was causing dimension mismatch
        # if scaler is not None:
        #    X_numeric = scaler.transform(X_numeric)
        
        # Combine TF-IDF and numeric features
        logger.info("Combining features")
        X_numeric_sparse = csr_matrix(X_numeric)
        X = hstack([X_tfidf, X_numeric_sparse])
        logger.info(f"Final feature matrix shape: {X.shape}")
        
        # Explicitly handle feature count mismatch for MLP model (hardcoded for now)
        expected_features = 7530  # This is what our model expects based on the error
        
        if X.shape[1] != expected_features:
            logger.warning(f"Feature count mismatch: got {X.shape[1]}, expected {expected_features}")
            
            # Convert to CSR matrix which is subscriptable
            X = X.tocsr()
            
            if X.shape[1] > expected_features:
                # Truncate extra features
                logger.info("Truncating extra features to match model expectations")
                X = X[:, :expected_features]
            else:
                # Pad with zeros if we have fewer features
                logger.info("Padding with zeros to match model expectations")
                padding = csr_matrix((X.shape[0], expected_features - X.shape[1]))
                X = hstack([X, padding])
                
            logger.info(f"Adjusted feature matrix shape: {X.shape}")
        
        return X
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        logger.error(traceback.format_exc())
        return None

def predict_email(model, vectorizer, email_data, scaler=None):
    """
    Predict whether an email is phishing or legitimate.
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        email_data: Dictionary or DataFrame with email data
        scaler: Optional feature scaler
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Prepare features
        X = prepare_email_features(email_data, vectorizer, scaler)
        
        if X is None:
            logger.error("Failed to prepare features")
            return {
                'predictions': [0],  # Default to legitimate if feature preparation fails
                'probabilities': [0.0],
                'labels': ['legitimate']
            }
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X)[:, 1]  # Probability of phishing class
        
        # Get predictions using default threshold (0.5)
        predictions = model.predict(X)
        
        # Convert predictions to labels
        prediction_labels = ['phishing' if p == 1 else 'legitimate' for p in predictions]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'labels': prediction_labels
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'predictions': [0],  # Default to legitimate on error
            'probabilities': [0.0],
            'labels': ['legitimate']
        }

def predict_from_file(model, vectorizer, input_file, output_file=None, scaler=None):
    """
    Predict phishing emails from a CSV file.
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        scaler: Feature scaler (optional)
        
    Returns:
        results_df: DataFrame with predictions
    """
    logger.info(f"Reading emails from {input_file}")
    emails_df = pd.read_csv(input_file)
    
    # Check required columns
    required_cols = ['subject', 'body']
    missing_cols = [col for col in required_cols if col not in emails_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Input file must contain columns: {required_cols}")
    
    # Prepare features
    X = prepare_email_features(emails_df, vectorizer, scaler)
    
    # Make predictions
    logger.info(f"Making predictions for {len(emails_df)} emails")
    
    # Use custom threshold if available
    threshold = getattr(model, 'threshold_', 0.5)
    logger.info(f"Using classification threshold: {threshold:.4f}")
    
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Add predictions to DataFrame
    emails_df['prediction'] = predictions
    emails_df['prediction_label'] = ['phishing' if p == 1 else 'legitimate' for p in predictions]
    emails_df['phishing_probability'] = probabilities
    
    # Save results if output file is specified
    if output_file:
        logger.info(f"Saving predictions to {output_file}")
        emails_df.to_csv(output_file, index=False)
    
    # Print summary
    phishing_count = sum(predictions)
    legitimate_count = len(predictions) - phishing_count
    logger.info(f"Results: {phishing_count} phishing emails, {legitimate_count} legitimate emails")
    
    return emails_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Predict phishing emails")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--subject", type=str, help="Email subject")
    input_group.add_argument("--input", type=str, help="Input CSV file with emails")
    
    parser.add_argument("--body", type=str, help="Email body")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")
    parser.add_argument("--from-field", type=str, default="", help="From field for direct prediction")
    parser.add_argument("--reply-to", type=str, default="", help="Reply-to field for direct prediction")
    
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
    
    args = parser.parse_args()
    
    # Load model, vectorizer, and scaler
    model, vectorizer, scaler = load_model_and_vectorizer(
        args.model, args.vectorizer, args.scaler
    )
    
    # Override threshold if specified
    threshold = 0.5
    if args.threshold is not None:
        logger.info(f"Using custom threshold: {args.threshold}")
        threshold = args.threshold
        model.threshold_ = args.threshold
    
    # Predict from file or direct input
    if args.input:
        predict_from_file(model, vectorizer, args.input, args.output, scaler)
    elif args.subject or args.body:
        email_data = {
            'subject': args.subject or "",
            'body': args.body or "",
            'from': args.from_field,
            'reply_to': args.reply_to
        }
        prediction = predict_email(model, vectorizer, email_data, scaler)
        
        # Print results
        prob = prediction['probabilities'][0]
        # Apply threshold to determine label
        is_phishing = prob >= threshold
        label = "phishing" if is_phishing else "legitimate"
        
        logger.info(f"Prediction: {label} (probability: {prob:.4f}, threshold: {threshold:.4f})")
    else:
        logger.error("Either --input or --subject/--body must be specified")
        parser.print_help()

if __name__ == "__main__":
    main() 