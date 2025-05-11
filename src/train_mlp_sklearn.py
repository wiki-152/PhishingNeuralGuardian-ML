#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the scikit-learn MLP model for phishing email detection.
"""

import pandas as pd
import numpy as np
import pickle
import time
import logging
import traceback
import gc
import os
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import threading
import queue
import multiprocessing
from joblib import parallel_backend
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import re

# Fix the import to use relative import
try:
    from features import make_features
    from model_builder_sklearn import build_mlp
except ImportError:
    # Try relative import if running from a different directory
    from src.features import make_features
    from src.model_builder_sklearn import build_mlp

# Configure logging to console for immediate feedback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("training.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# Set number of CPU cores to use
N_JOBS = max(1, multiprocessing.cpu_count() - 1)
logger.info(f"Using {N_JOBS} CPU cores for parallel processing")

class ProgressBarCallback:
    """
    Custom callback to show progress during MLPClassifier training.
    
    Uses a separate thread to update a tqdm progress bar during training.
    """
    def __init__(self, max_iter, update_interval=0.2):
        self.max_iter = max_iter
        self.update_interval = update_interval
        self.pbar = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.thread = None
        self.current_iter = 0
        
    def __enter__(self):
        # Create progress bar with more visible format
        self.pbar = tqdm(
            total=self.max_iter, 
            desc="Training model", 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor_progress)
        self.thread.daemon = True
        self.thread.start()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        # Close progress bar
        if self.pbar:
            self.pbar.close()
    
    def _monitor_progress(self):
        """Monitor progress and update the progress bar."""
        last_iter = 0
        
        while not self.stop_event.is_set():
            try:
                # Check if there's a new iteration count
                if not self.progress_queue.empty():
                    current_iter = self.progress_queue.get()
                    
                    # Update progress bar
                    if current_iter > last_iter:
                        self.pbar.update(current_iter - last_iter)
                        last_iter = current_iter
                        
                        # Print status every 10 iterations
                        if current_iter % 10 == 0:
                            print(f"Iteration {current_iter}/{self.max_iter} completed")
                        
                        # If we've reached max_iter, stop monitoring
                        if current_iter >= self.max_iter:
                            break
            except Exception as e:
                logger.error(f"Error in progress monitoring: {e}")
                
            # Sleep for a short time
            time.sleep(self.update_interval)
    
    def update(self, n_iter):
        """Update the progress with the current iteration count."""
        self.progress_queue.put(n_iter)

class SimpleProgressMonitor:
    """
    A simple progress monitor for scikit-learn's MLPClassifier.
    """
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.pbar = None
        self.current_iter = 0
        
    def __enter__(self):
        self.pbar = tqdm(
            total=self.max_iter,
            desc="Training model",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
            
    def update(self, increment=1):
        self.current_iter += increment
        self.pbar.update(increment)
        
        # Print status every 10 iterations
        if self.current_iter % 10 == 0:
            print(f"Iteration {self.current_iter}/{self.max_iter} completed")

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and save confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
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
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        logger.debug(traceback.format_exc())

def clean_data(df):
    """
    Clean the dataset by handling missing values, removing duplicates,
    and filtering out problematic entries.
    """
    logger.info("Cleaning dataset...")
    initial_size = len(df)
    
    # Handle missing values
    df['subject'] = df['subject'].fillna("[No Subject]")
    df['body'] = df['body'].fillna("[No Body]")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['subject', 'body']).reset_index(drop=True)
    
    # Filter out rows with extremely short content
    min_content_length = 10  # Minimum characters
    df = df[df['body'].str.len() >= min_content_length].reset_index(drop=True)
    
    # Filter out rows with suspicious patterns (like all NaN values)
    df = df[~df['subject'].str.contains('nan', case=False, na=False)].reset_index(drop=True)
    
    # Remove emails with nonsensical content (e.g., random strings)
    def is_meaningful(text):
        # Check if text has a reasonable word/character ratio
        words = re.findall(r'\w+', str(text))
        if not words:
            return False
        avg_word_length = sum(len(word) for word in words) / len(words)
        return 2 <= avg_word_length <= 15  # Reasonable average word length
    
    df = df[df['body'].apply(is_meaningful)].reset_index(drop=True)
    
    logger.info(f"Removed {initial_size - len(df)} problematic entries ({(initial_size - len(df))/initial_size*100:.2f}%)")
    return df

def balance_dataset(X, y, strategy='combined'):
    """
    Balance the dataset using various strategies.
    
    Args:
        X: Feature matrix
        y: Target labels
        strategy: Balancing strategy ('oversample', 'undersample', or 'combined')
        
    Returns:
        X_balanced, y_balanced: Balanced feature matrix and target labels
    """
    logger.info(f"Balancing dataset using strategy: {strategy}")
    
    # Check if X is sparse
    is_sparse = hasattr(X, 'tocsr')
    if is_sparse and not isinstance(X, np.ndarray):
        logger.info("Feature matrix is sparse, ensuring CSR format for indexing")
        X = X.tocsr()
    
    # Safety check - ensure y only contains valid binary labels
    unique_labels = np.unique(y)
    if not np.all(np.isin(unique_labels, [0, 1])):
        logger.warning(f"Unexpected labels in dataset: {unique_labels}. Filtering to keep only 0 and 1.")
        valid_mask = np.isin(y, [0, 1])
        X = X[valid_mask]
        y = y[valid_mask]
        if len(y) == 0:
            logger.error("No valid samples with labels 0 or 1 remain after filtering")
            raise ValueError("No valid samples with binary labels found in dataset")
    
    # Convert y to integer type if it's float
    if y.dtype == np.float64 or y.dtype == np.float32:
        logger.info("Converting float labels to integers")
        y = np.round(y).astype(int)
    
    # Get class distribution
    class_counts = np.bincount(y)
    
    # Handle case where only one class is present
    if len(class_counts) == 1:
        logger.warning(f"Only one class ({np.unique(y)[0]}) present in the dataset")
        return X, y
        
    # Normal processing for binary case
    majority_class = np.argmax(class_counts)
    minority_class = 1 - majority_class
    
    # Calculate imbalance ratio
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if strategy == 'oversample':
        # Use SMOTE to oversample the minority class
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
    elif strategy == 'undersample':
        # Undersample the majority class
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
        
    elif strategy == 'combined':
        # Combined approach: First undersample majority class, then oversample minority
        # This works better for extremely imbalanced datasets
        
        # If the imbalance is not severe, adjust the undersampling ratio
        if imbalance_ratio < 1.5:
            # No need for aggressive balancing, just do slight adjustment
            logger.info("Class imbalance is mild, applying gentle balancing")
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        else:
            # For more imbalanced datasets, use a two-step approach
            # Calculate a safe undersampling ratio that won't cause issues
            safe_ratio = max(0.6, 1.0 / imbalance_ratio)
            logger.info(f"Using undersampling ratio of {safe_ratio:.2f}")
            
            # First, undersample the majority class but not too aggressively
            sampling_strategy = {majority_class: int(class_counts[majority_class] * safe_ratio), 
                                minority_class: class_counts[minority_class]}
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_temp, y_temp = rus.fit_resample(X, y)
            
            # Then apply SMOTE to balance the classes completely
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_temp, y_temp)
    else:
        # No balancing, just return original data
        logger.info("No balancing applied")
        return X, y
    
    logger.info(f"Original class distribution: {np.bincount(y)}")
    logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
    
    return X_balanced, y_balanced

class MLPProgressCallback:
    """
    Custom callback for MLPClassifier to show progress during training.
    
    This class is kept for compatibility but is no longer used directly by MLPClassifier.
    Instead, we use tqdm with partial_fit in the training loop.
    """
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.current_iter = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.pbar = None
        
    def __call__(self, locals_dict):
        """Called after each iteration of the MLPClassifier."""
        # This method is kept for compatibility but is no longer used
        pass

def main(use_sample=False, max_features=5000, data_path=None, balance_strategy='combined'):
    """
    Main training function.
    
    Args:
        use_sample: Whether to use a sample of the data
        max_features: Maximum number of TF-IDF features
        data_path: Path to the dataset
        balance_strategy: Strategy for balancing the dataset
    """
    try:
        # Set up proper paths for model saving
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"Models will be saved to: {models_dir}")
        
        logger.info("Starting MLP training process")
        
        # Set data path
        if not data_path:
            data_path = "../data/processed/all_combined.csv"
        
        # Load and prepare data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Sample dataset if it's too large
        if len(df) > 150000 and use_sample:
            logger.info(f"Dataset is large ({len(df)} records). Taking a random sample of 10,000 records.")
            df = df.sample(n=10000, random_state=42)
            logger.info(f"Sampled dataset size: {len(df)}")
        
        # Create features
        logger.info("Creating features...")
        X, y, vectorizer, _ = make_features(df, max_tfidf=max_features)
        
        # Check matrix type
        is_sparse = hasattr(X, 'tocsr')
        if is_sparse:
            logger.info("Feature matrix is sparse, converting to CSR format for indexing")
            X = X.tocsr()
        
        # Clean y - remove rows with NaN values or negative values
        nan_mask = np.isnan(y)
        if np.any(nan_mask):
            valid_indices = ~nan_mask
            logger.warning(f"Removing {np.sum(nan_mask)} rows with NaN labels")
            if is_sparse:
                X = X[valid_indices]
            else:
                X = X[valid_indices]
            y = y[valid_indices]
        
        # Convert y to integers if needed
        if y.dtype == np.float64 or y.dtype == np.float32:
            logger.info("Converting target labels to integers")
            y = np.round(y).astype(int)
            
        # Ensure y has only valid values (0 or 1)
        invalid_mask = ~((y == 0) | (y == 1))
        if np.any(invalid_mask):
            valid_indices = ~invalid_mask
            logger.warning(f"Removing {np.sum(invalid_mask)} rows with invalid labels (not 0 or 1)")
            if is_sparse:
                X = X[valid_indices]
            else:
                X = X[valid_indices]
            y = y[valid_indices]
        
        # Split data
        logger.info("Splitting data into train, validation, and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Balance training data
        logger.info(f"Balancing dataset using strategy: {balance_strategy}")
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, strategy=balance_strategy)
        
        # Build and train model
        logger.info("Building and training model...")
        model = build_mlp(
            hidden_layers=(256, 128, 64),
            max_iter=75,
            threshold=0.5
        )
        
        # Train with progress monitoring
        with ProgressBarCallback(max_iter=75) as progress:
            model.fit(X_train_balanced, y_train_balanced)
        
        # Find optimal threshold
        logger.info("Finding optimal classification threshold...")
        y_val_prob = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_val_pred = (y_val_prob >= threshold).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        model.threshold = best_threshold
        
        # Evaluate on validation set
        logger.info("Evaluating model on validation set...")
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation AUC: {val_auc:.4f}")
        
        # Print classification report
        print("\nClassification Report (Validation):")
        print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Phishing']))
        
        # Generate confusion matrix
        logger.info("Generating validation confusion matrix...")
        val_confusion_matrix_path = os.path.join(models_dir, "val_confusion_matrix.png")
        plot_confusion_matrix(y_val, y_val_pred, save_path=val_confusion_matrix_path)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        
        # Print classification report
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Phishing']))
        
        # Generate confusion matrix
        logger.info("Generating test confusion matrix...")
        test_confusion_matrix_path = os.path.join(models_dir, "test_confusion_matrix.png")
        plot_confusion_matrix(y_test, y_test_pred, save_path=test_confusion_matrix_path)
        
        # Save model and vectorizer
        logger.info("Saving model and vectorizer...")
        model_path = os.path.join(models_dir, "mlp_sklearn.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        vectorizer_path = os.path.join(models_dir, "tfidf.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Vectorizer saved to: {vectorizer_path}")
        
        return {
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc
        }
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        logger.error(traceback.format_exc())
        raise

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MLP model for phishing detection")
    
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Force using the full dataset for training (default behavior)"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features to use"
    )
    
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=N_JOBS,
        help="Number of CPU cores to use"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the dataset file"
    )
    
    parser.add_argument(
        "--balance",
        type=str,
        choices=['none', 'oversample', 'undersample', 'combined'],
        default='combined',
        help="Data balancing strategy"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    
    args = parse_args()
    
    # Update CPU cores if specified
    if args.cpu_cores:
        N_JOBS = args.cpu_cores
        print(f"Using {N_JOBS} CPU cores for training")
    
    # Use specified data path if provided
    data_path = args.data_path if args.data_path else None
        
    # Always use the full dataset, never the sample
    main(use_sample=False, max_features=args.max_features, data_path=data_path, balance_strategy=args.balance) 