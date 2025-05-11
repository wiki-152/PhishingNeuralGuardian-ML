#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and evaluate the baseline MLP model for phishing email detection.
This script loads the processed data, creates features, trains the model,
evaluates performance, and saves the results.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

from features import make_features
from model import PhishingClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train phishing detection model')
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/all_combined.csv',
        help='Path to the processed data CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/baseline_mlp',
        help='Directory to save the model and results'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--hidden-layers',
        type=str,
        default='128,64,32',
        help='Comma-separated list of hidden layer sizes'
    )
    
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.3,
        help='Dropout rate for regularization'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of epochs for training'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Patience for early stopping'
    )
    
    return parser.parse_args()

def save_results(metrics, output_dir):
    """Save evaluation metrics and plots."""
    output_dir = Path(output_dir)
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics as JSON
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # Save confusion matrix as CSV
    pd.DataFrame(
        metrics['confusion_matrix'],
        columns=['Predicted Legitimate', 'Predicted Phishing'],
        index=['True Legitimate', 'True Phishing']
    ).to_csv(results_dir / 'confusion_matrix.csv')
    
    # Save ROC curve data
    pd.DataFrame({
        'fpr': metrics['fpr'],
        'tpr': metrics['tpr']
    }).to_csv(results_dir / 'roc_curve.csv', index=False)
    
    logger.info(f"Results saved to {results_dir}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    try:
        df = pd.read_csv(args.data_path)
        logger.info(f"Loaded {len(df)} rows from {args.data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Filter to only use rows with known labels
    df = df[df['label'].isin(['phishing', 'legitimate'])].reset_index(drop=True)
    logger.info(f"Using {len(df)} rows with known labels")
    
    # Log class distribution
    class_counts = df['label'].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    # Create features
    logger.info("Creating features")
    features_dir = output_dir / 'features'
    X, y, vectorizer, numeric_cols = make_features(df, output_dir=features_dir)
    logger.info(f"Created feature matrix with shape {X.shape}")
    
    # Split data
    logger.info(f"Splitting data with test_size={args.test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Parse hidden layers
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    
    # Initialize model
    logger.info(f"Initializing model with hidden layers {hidden_layers}")
    classifier = PhishingClassifier(
        hidden_layers=hidden_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        random_state=args.random_state
    )
    
    # Train model
    logger.info("Training model")
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model")
    metrics = classifier.evaluate(X_test, y_test)
    
    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    classifier.save(output_dir)
    
    # Save results
    save_results(metrics, output_dir)
    
    # Create and save plots
    logger.info("Creating plots")
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Training history plot
    history_fig = classifier.plot_training_history()
    history_fig.savefig(plots_dir / 'training_history.png')
    
    # Confusion matrix plot
    cm_fig = classifier.plot_confusion_matrix(metrics['confusion_matrix'])
    cm_fig.savefig(plots_dir / 'confusion_matrix.png')
    
    # ROC curve plot
    roc_fig = classifier.plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
    roc_fig.savefig(plots_dir / 'roc_curve.png')
    
    logger.info(f"Plots saved to {plots_dir}")
    logger.info("Training and evaluation complete")

if __name__ == "__main__":
    main() 