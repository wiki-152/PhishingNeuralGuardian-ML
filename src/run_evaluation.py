#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run model evaluation.
"""

import os
import sys
from pathlib import Path
from evaluate_model import evaluate_model
from predict_phishing import load_model_and_vectorizer

def main():
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer(
        model_path="models/mlp_sklearn.pkl",
        vectorizer_path="models/tfidf.pkl"
    )
    
    # Create output directory
    output_dir = Path("results/evaluation")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run evaluation
    print("Running evaluation...")
    metrics = evaluate_model(
        model=model,
        vectorizer=vectorizer,
        test_data_path="data/processed/test_emails.csv",
        output_dir=output_dir
    )
    
    if metrics:
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main() 