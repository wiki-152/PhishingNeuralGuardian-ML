#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify the phishing detection model works correctly.
"""

import os
import sys
import logging
from pathlib import Path
from predict_phishing import predict_email, load_model_and_vectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test the model with simple examples."""
    # Define paths to model files
    model_path = "../models/mlp_sklearn.pkl"
    vectorizer_path = "../models/tfidf.pkl"
    scaler_path = "../models/scaler.pkl"
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model files not found. Please train the model first.")
        print(f"Expected model at: {model_path}")
        print(f"Expected vectorizer at: {vectorizer_path}")
        return 1
    
    # Load model and vectorizer
    try:
        model, vectorizer, scaler = load_model_and_vectorizer(model_path, vectorizer_path, scaler_path)
        print(f"Model loaded successfully.")
        
        # Get the model's threshold
        threshold = getattr(model, 'threshold_', 0.5)
        print(f"Classification threshold: {threshold:.4f}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Define test examples
    test_examples = [
        {
            "name": "Obvious phishing email",
            "subject": "URGENT: Your account has been compromised",
            "body": "Dear user, your account has been compromised. Click here to verify your information: http://suspicious-link.com",
            "from": "security@g00gle.com",
            "reply_to": "hacker@evil.com"
        },
        {
            "name": "Legitimate email",
            "subject": "Meeting reminder for tomorrow",
            "body": "Hi team, just a reminder that we have a meeting scheduled for tomorrow at 10am. Please prepare your weekly reports. Thanks, Manager",
            "from": "manager@company.com",
            "reply_to": "manager@company.com"
        },
        {
            "name": "Ambiguous email",
            "subject": "Your recent purchase",
            "body": "Thank you for your recent purchase. Your order has been processed and will be shipped soon. To track your order, please log in to your account.",
            "from": "orders@amazon-store.net",
            "reply_to": "no-reply@amazon-store.net"
        }
    ]
    
    # Test each example
    print("\n" + "=" * 80)
    print("TESTING MODEL WITH EXAMPLES")
    print("=" * 80)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}: {example['name']}")
        print(f"Subject: {example['subject']}")
        print(f"From: {example['from']}")
        print(f"Reply-To: {example['reply_to']}")
        print(f"Body: {example['body'][:100]}...")
        
        # Make prediction
        prediction, probability = predict_email(model, vectorizer, example, scaler)
        
        # Print result
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        print(f"\nPrediction: {result}")
        print(f"Confidence: {probability:.4f} (threshold: {threshold:.4f})")
        print("-" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 