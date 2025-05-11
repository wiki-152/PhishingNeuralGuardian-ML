#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to find the optimal classification threshold for the phishing detection model.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix the import to use relative import
try:
    from features import make_features
    from predict_phishing import load_model_and_vectorizer, prepare_email_features
except ImportError:
    # Try relative import if running from a different directory
    from src.features import make_features
    from src.predict_phishing import load_model_and_vectorizer, prepare_email_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("threshold_optimization.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

def evaluate_threshold(y_true, y_prob, threshold):
    """
    Evaluate model performance at a specific threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with performance metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'specificity': recall_score(y_true, y_pred, pos_label=0),
        'tp': ((y_pred == 1) & (y_true == 1)).sum(),
        'fp': ((y_pred == 1) & (y_true == 0)).sum(),
        'tn': ((y_pred == 0) & (y_true == 0)).sum(),
        'fn': ((y_pred == 0) & (y_true == 1)).sum()
    }

def find_optimal_threshold(model, X_val, y_val, metric='f1'):
    """
    Find the optimal classification threshold based on validation data.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        best_threshold: Optimal threshold
        metrics: DataFrame with metrics for all tested thresholds
    """
    logger.info(f"Finding optimal threshold based on {metric} score")
    
    # Get predicted probabilities
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    results = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        metrics = evaluate_threshold(y_val, y_val_prob, threshold)
        results.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Find best threshold
    best_idx = metrics_df[metric].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'threshold']
    best_metrics = metrics_df.loc[best_idx]
    
    logger.info(f"Best threshold: {best_threshold:.4f}")
    logger.info(f"Best {metric} score: {best_metrics[metric]:.4f}")
    logger.info(f"Metrics at best threshold: accuracy={best_metrics['accuracy']:.4f}, "
                f"precision={best_metrics['precision']:.4f}, recall={best_metrics['recall']:.4f}")
    
    return best_threshold, metrics_df

def plot_threshold_metrics(metrics_df, save_path=None):
    """
    Plot metrics across different thresholds.
    
    Args:
        metrics_df: DataFrame with metrics for different thresholds
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy')
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score')
    plt.plot(metrics_df['threshold'], metrics_df['specificity'], label='Specificity')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Classification Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Threshold metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curve plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Precision-Recall curve plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize classification threshold for phishing detection model")
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to validation data CSV file"
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
        "--metric",
        type=str,
        choices=['f1', 'accuracy', 'precision', 'recall'],
        default='f1',
        help="Metric to optimize"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Directory to save plots and results"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model, vectorizer, and scaler
    logger.info("Loading model and vectorizer")
    model, vectorizer, scaler = load_model_and_vectorizer(
        args.model, args.vectorizer, args.scaler
    )
    
    # Load validation data
    logger.info(f"Loading validation data from {args.data}")
    try:
        val_data = pd.read_csv(args.data)
        
        # Check required columns
        required_cols = ['subject', 'body', 'label']
        missing_cols = [col for col in required_cols if col not in val_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Convert labels to numeric
        val_data['label_numeric'] = val_data['label'].map({'legitimate': 0, 'phishing': 1})
        
        # Prepare features
        logger.info("Preparing features")
        X_val = prepare_email_features(val_data, vectorizer, scaler)
        y_val = val_data['label_numeric'].values
        
        # Find optimal threshold
        logger.info(f"Finding optimal threshold based on {args.metric}")
        best_threshold, metrics_df = find_optimal_threshold(model, X_val, y_val, args.metric)
        
        # Save metrics to CSV
        metrics_path = output_dir / "threshold_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Threshold metrics saved to {metrics_path}")
        
        # Plot metrics
        logger.info("Generating plots")
        plot_threshold_metrics(metrics_df, save_path=output_dir / "threshold_metrics.png")
        
        # Get predicted probabilities
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        # Plot ROC curve
        plot_roc_curve(y_val, y_val_prob, save_path=output_dir / "roc_curve.png")
        
        # Plot precision-recall curve
        plot_precision_recall_curve(y_val, y_val_prob, save_path=output_dir / "pr_curve.png")
        
        # Save best threshold
        with open(output_dir / "optimal_threshold.txt", "w") as f:
            f.write(f"Optimal threshold: {best_threshold:.4f}\n")
            f.write(f"Optimized for: {args.metric}\n")
            f.write(f"Accuracy: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'accuracy'].values[0]:.4f}\n")
            f.write(f"Precision: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'precision'].values[0]:.4f}\n")
            f.write(f"Recall: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'recall'].values[0]:.4f}\n")
            f.write(f"F1 Score: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'f1'].values[0]:.4f}\n")
        
        logger.info(f"Optimal threshold ({best_threshold:.4f}) saved to {output_dir / 'optimal_threshold.txt'}")
        
        # Update model with optimal threshold
        model.threshold_ = best_threshold
        
        # Save updated model
        with open(args.model, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Updated model saved with optimal threshold")
        
        print("\n" + "=" * 80)
        print(f"OPTIMAL THRESHOLD: {best_threshold:.4f}")
        print(f"Optimized for: {args.metric}")
        print(f"Accuracy: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'accuracy'].values[0]:.4f}")
        print(f"Precision: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'precision'].values[0]:.4f}")
        print(f"Recall: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'recall'].values[0]:.4f}")
        print(f"F1 Score: {metrics_df.loc[metrics_df['threshold'] == best_threshold, 'f1'].values[0]:.4f}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 