#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the MLP model for phishing email detection.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from features import make_features
from model_builder import build_mlp

def main():
    # Set paths
    data_path = Path("data/processed/all_combined.csv")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} emails")
    
    # Filter to only use rows with known labels
    df = df[df['label'].isin(['phishing', 'legitimate'])].reset_index(drop=True)
    print(f"Using {len(df)} rows with known labels")
    
    # Create features
    print("Creating features...")
    X, y, vectorizer, _ = make_features(df)
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data into train, validation, and test sets (70-15-15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert sparse matrices to dense arrays for TensorFlow
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()
    
    # Build model
    print("Building model...")
    model = build_mlp(X_train.shape[1])
    model.summary()
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=2,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy, auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    
    # Test set evaluation
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    model.save(model_dir / "mlp.h5")
    with open(model_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_dir / 'mlp.h5'}")
    print(f"Vectorizer saved to {model_dir / 'tfidf.pkl'}")

if __name__ == "__main__":
    main() 