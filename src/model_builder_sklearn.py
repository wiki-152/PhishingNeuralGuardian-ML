#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build and configure the MLP model for phishing detection.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)

class ThresholdMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper around MLPClassifier that adds threshold-based prediction.
    This class is pickle-friendly and maintains all the functionality of MLPClassifier.
    """
    def __init__(self, hidden_layers=(256, 128, 64), max_iter=75, threshold=0.5):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.threshold = threshold
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    
    def fit(self, X, y):
        """Fit the model to the data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels using the threshold."""
        return self.predict_with_threshold(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def predict_with_threshold(self, X, threshold=None):
        """Predict class labels using a custom threshold."""
        if threshold is None:
            threshold = self.threshold
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """Return the accuracy score on the given test data and labels."""
        return self.model.score(X, y)

def build_mlp(hidden_layers=(256, 128, 64), max_iter=75, threshold=0.5):
    """
    Build and configure an MLP model for phishing detection.
    
    Args:
        hidden_layers: Tuple of integers specifying the number of neurons in each hidden layer
        max_iter: Maximum number of iterations for training
        threshold: Classification threshold for phishing detection
        
    Returns:
        A configured ThresholdMLPClassifier instance
    """
    logger.info(f"Building MLP model with architecture: {hidden_layers}")
    logger.info(f"Max iterations: {max_iter}")
    logger.info(f"Initial threshold: {threshold}")
    
    model = ThresholdMLPClassifier(
        hidden_layers=hidden_layers,
        max_iter=max_iter,
        threshold=threshold
    )
    
    return model 