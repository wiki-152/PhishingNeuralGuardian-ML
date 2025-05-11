#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify model saving logic using a dummy dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our model builder
from model_builder_sklearn import build_mlp

def create_dummy_data(n_samples=1000):
    """Create a small dummy dataset for testing."""
    # Create random features
    X = np.random.randn(n_samples, 100)
    
    # Create binary labels with some pattern
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y

def test_model_save():
    """Test the model saving functionality."""
    try:
        logger.info("Creating dummy dataset...")
        X, y = create_dummy_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        logger.info("Building model...")
        model = build_mlp(
            hidden_layers=(32, 16),  # Smaller architecture for testing
            max_iter=5,  # Few iterations for quick testing
            threshold=0.5
        )
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Test predictions
        logger.info("Testing predictions...")
        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        logger.info("Saving model...")
        model_path = Path("../models/test_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Load model
        logger.info("Loading model...")
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        
        # Verify loaded model
        logger.info("Verifying loaded model...")
        loaded_predictions = loaded_model.predict(X_test)
        loaded_accuracy = loaded_model.score(X_test, y_test)
        
        # Compare results
        predictions_match = np.array_equal(predictions, loaded_predictions)
        accuracy_match = abs(accuracy - loaded_accuracy) < 1e-10
        
        logger.info(f"Original accuracy: {accuracy:.4f}")
        logger.info(f"Loaded model accuracy: {loaded_accuracy:.4f}")
        logger.info(f"Predictions match: {predictions_match}")
        logger.info(f"Accuracy matches: {accuracy_match}")
        
        # Test threshold prediction
        logger.info("Testing threshold prediction...")
        threshold_predictions = loaded_model.predict_with_threshold(X_test, threshold=0.7)
        logger.info(f"Threshold predictions shape: {threshold_predictions.shape}")
        
        return {
            'predictions_match': predictions_match,
            'accuracy_match': accuracy_match,
            'original_accuracy': accuracy,
            'loaded_accuracy': loaded_accuracy
        }
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        raise

if __name__ == "__main__":
    results = test_model_save()
    print("\nTest Results:")
    print(f"Predictions match: {results['predictions_match']}")
    print(f"Accuracy matches: {results['accuracy_match']}")
    print(f"Original accuracy: {results['original_accuracy']:.4f}")
    print(f"Loaded model accuracy: {results['loaded_accuracy']:.4f}") 