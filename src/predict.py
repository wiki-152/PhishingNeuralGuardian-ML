#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make predictions on new emails using the trained phishing detection model.
This script can process individual email files or directories of emails.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import email
from email import policy
import json
import sys

from model import PhishingClassifier
from features import count_urls, count_urgency_words, detect_spoofing
from data_loader import load_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict phishing emails')
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to email file or directory containing emails'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/baseline_mlp',
        help='Directory containing the trained model'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (0.0-1.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save prediction results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about each prediction'
    )
    
    return parser.parse_args()

def load_model_and_preprocessors(model_dir):
    """
    Load the trained model and preprocessing components.
    
    Args:
        model_dir: Directory containing the model and preprocessors
        
    Returns:
        classifier: Trained PhishingClassifier
        vectorizer: Fitted TF-IDF vectorizer
        scaler: Fitted StandardScaler
    """
    model_dir = Path(model_dir)
    
    # Load the classifier
    classifier = PhishingClassifier.load(model_dir)
    
    # Load the vectorizer and scaler
    features_dir = model_dir / 'features'
    vectorizer = joblib.load(features_dir / 'tfidf_vectorizer.joblib')
    scaler = joblib.load(features_dir / 'numeric_scaler.joblib')
    
    return classifier, vectorizer, scaler

def process_and_predict(input_path, classifier, vectorizer, scaler, threshold=0.5):
    """
    Process emails and make predictions.
    
    Args:
        input_path: Path to email file or directory
        classifier: Trained PhishingClassifier
        vectorizer: Fitted TF-IDF vectorizer
        scaler: Fitted StandardScaler
        threshold: Classification threshold
        
    Returns:
        DataFrame with predictions
    """
    # Load emails using the existing data loader
    df = load_messages([input_path])
    
    if df.empty:
        logger.error(f"No valid emails found at {input_path}")
        return pd.DataFrame()
    
    # Prepare features
    # Combine subject and body for text features
    df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    
    # Create TF-IDF features
    X_tfidf = vectorizer.transform(df['text'])
    
    # Create numeric features
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['url_count'] = df['text'].apply(count_urls)
    
    # Sentiment analysis (using imported function)
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
        df['sentiment_compound'] = df['text'].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
    except Exception as e:
        logger.warning(f"Error in sentiment analysis: {str(e)}")
        df['sentiment_compound'] = 0.0
    
    # Urgency heuristic
    df['urgency_hits'] = df['text'].apply(count_urgency_words)
    
    # Spoofing detection
    df['spoof_flag'] = df.apply(
        lambda row: detect_spoofing(row['from'], row.get('reply_to')), 
        axis=1
    )
    
    # List of numeric columns
    numeric_cols = ['word_count', 'url_count', 'sentiment_compound', 'urgency_hits', 'spoof_flag']
    
    # Scale numeric features
    X_numeric = scaler.transform(df[numeric_cols])
    
    # Combine TF-IDF and numeric features
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_numeric])
    
    # Make predictions
    df['probability'] = classifier.predict_proba(X)
    df['prediction'] = classifier.predict(X, threshold=threshold)
    df['prediction_label'] = df['prediction'].map({1: 'phishing', 0: 'legitimate'})
    
    # Add confidence level categories
    df['confidence'] = df['probability'].apply(
        lambda p: 'high' if abs(p - 0.5) > 0.4 else 
                 ('medium' if abs(p - 0.5) > 0.2 else 'low')
    )
    
    return df

def format_results(df, verbose=False):
    """Format prediction results for display."""
    results = []
    
    for _, row in df.iterrows():
        result = {
            'from': row['from'],
            'subject': row['subject'],
            'prediction': row['prediction_label'],
            'probability': float(row['probability']),
            'confidence': row['confidence']
        }
        
        if verbose:
            result.update({
                'url_count': int(row['url_count']),
                'word_count': int(row['word_count']),
                'urgency_hits': int(row['urgency_hits']),
                'sentiment_compound': float(row['sentiment_compound']),
                'spoof_flag': int(row['spoof_flag'])
            })
        
        results.append(result)
    
    return results

def main():
    """Main prediction function."""
    args = parse_args()
    
    # Load model and preprocessors
    try:
        logger.info(f"Loading model from {args.model_dir}")
        classifier, vectorizer, scaler = load_model_and_preprocessors(args.model_dir)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return 1
    
    # Process emails and make predictions
    try:
        logger.info(f"Processing emails from {args.input_path}")
        df = process_and_predict(args.input_path, classifier, vectorizer, scaler, args.threshold)
        
        if df.empty:
            logger.error("No valid predictions made")
            return 1
            
        logger.info(f"Made predictions for {len(df)} emails")
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return 1
    
    # Format results
    results = format_results(df, args.verbose)
    
    # Display results
    for i, result in enumerate(results):
        phish_prob = result['probability']
        confidence = result['confidence']
        
        print(f"\nEmail {i+1}:")
        print(f"From: {result['from']}")
        print(f"Subject: {result['subject']}")
        print(f"Prediction: {result['prediction']} (Probability: {phish_prob:.2f}, Confidence: {confidence})")
        
        if args.verbose:
            print("Features:")
            print(f"  - URLs: {result['url_count']}")
            print(f"  - Words: {result['word_count']}")
            print(f"  - Urgency terms: {result['urgency_hits']}")
            print(f"  - Sentiment: {result['sentiment_compound']:.2f}")
            print(f"  - Spoofing detected: {'Yes' if result['spoof_flag'] == 1 else 'No'}")
    
    # Save results if output path provided
    if args.output:
        try:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    # Summary
    phishing_count = sum(1 for r in results if r['prediction'] == 'phishing')
    legitimate_count = len(results) - phishing_count
    
    print("\nSummary:")
    print(f"Total emails: {len(results)}")
    print(f"Phishing: {phishing_count}")
    print(f"Legitimate: {legitimate_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 