#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate the trained phishing detection model on a test dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from predict_phishing import load_model_and_vectorizer, prepare_email_features, predict_email

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, labels, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    # Close the figure to free memory
    plt.close()

def evaluate_model(model, vectorizer, test_data_path, output_dir=None):
    """
    Evaluate the model on a test dataset.
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        test_data_path: Path to test data CSV
        output_dir: Directory to save evaluation results
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Check required columns
    required_cols = ['subject', 'body']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Test data must contain columns: {required_cols}")
    
    # Check if label column exists
    has_labels = 'label' in test_df.columns
    if not has_labels:
        logger.warning("No 'label' column found in test data. Will only make predictions.")
    else:
        # Convert labels to binary
        logger.info("Converting labels to binary format")
        test_df['true_label'] = test_df['label'].apply(
            lambda x: 1 if x.lower().strip() == 'phishing' else 0
        )
        logger.info(f"Label distribution: {test_df['true_label'].value_counts().to_dict()}")
    
    # Make predictions
    logger.info("Making predictions")
    predictions = predict_email(model, vectorizer, test_df)
    
    if predictions is None:
        logger.error("Failed to make predictions")
        return None
    
    # Add predictions to DataFrame
    test_df['prediction'] = predictions['predictions']
    test_df['prediction_label'] = predictions['labels']
    test_df['phishing_probability'] = predictions['probabilities']
    
    # Calculate metrics if labels are available
    metrics = {}
    if has_labels:
        logger.info("Calculating performance metrics")
        y_true = test_df['true_label']
        y_pred = test_df['prediction']
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['auc'] = roc_auc_score(y_true, predictions['probabilities'])
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Legitimate', 'Phishing'],
            output_dict=True
        )
        metrics['class_report'] = class_report
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Print metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Plot confusion matrix
        if output_dir:
            plot_confusion_matrix(
                cm, ['Legitimate', 'Phishing'], 
                save_path=output_dir / "confusion_matrix.png"
            )
        else:
            print("\nConfusion Matrix:")
            print(cm)
        
        # Add correct/incorrect column
        test_df['correct'] = test_df['prediction'] == test_df['true_label']
        
        # Print examples of misclassified emails
        misclassified = test_df[~test_df['correct']]
        if len(misclassified) > 0:
            logger.info(f"\nFound {len(misclassified)} misclassified emails:")
            for i, row in misclassified.iterrows():
                print(f"\nMisclassified Example {i+1}:")
                print(f"Subject: {row['subject']}")
                print(f"True label: {row['label']}")
                print(f"Predicted: {row['prediction_label']} (probability: {row['phishing_probability']:.4f})")
                print("-" * 50)
    
    # Save results if output directory is specified
    if output_dir:
        logger.info(f"Saving evaluation results to {output_dir}")
        
        # Save predictions
        test_df.to_csv(output_dir / "predictions.csv", index=False)
        
        # Save metrics
        if has_labels:
            pd.DataFrame({k: [v] for k, v in metrics.items() 
                        if k not in ['class_report', 'confusion_matrix']}).to_csv(
                output_dir / "metrics.csv", index=False
            )
    
    return metrics

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate phishing detection model")
    
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
        "--test-data",
        type=str,
        required=True,
        help="Path to test data CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Load model and vectorizer
    model, vectorizer, scaler = load_model_and_vectorizer(args.model, args.vectorizer)
    
    # Evaluate model
    evaluate_model(model, vectorizer, args.test_data, args.output_dir)

if __name__ == "__main__":
    main() 