#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train an improved phishing detection model with optimized parameters.
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
from train_mlp_sklearn import main as train_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("improved_training.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the improved training process."""
    parser = argparse.ArgumentParser(description="Train improved phishing detection model")
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the dataset file"
    )
    
    parser.add_argument(
        "--balance",
        type=str,
        choices=['none', 'oversample', 'undersample', 'combined'],
        default='none',  # Changed to none for initial testing
        help="Data balancing strategy"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=7500,  # Increased from default 5000
        help="Maximum number of TF-IDF features to use"
    )
    
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create necessary directories if they don't exist"
    )
    
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use a sample of the dataset (100,000 records) for faster training"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("IMPROVED PHISHING EMAIL DETECTION - TRAINING PIPELINE")
    print("=" * 80)
    
    logger.info("Starting improved training process")
    
    # Set data path
    data_path = args.data_path
    if not data_path:
        data_path = "../data/processed/all_combined.csv"
    
    # Create directories if needed
    if args.create_dirs:
        try:
            # Create data directories
            data_dir = Path("../data/processed")
            data_dir.mkdir(exist_ok=True, parents=True)
            
            # Create models directory
            models_dir = Path("../models")
            models_dir.mkdir(exist_ok=True, parents=True)
            
            # Ensure absolute paths are used
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            abs_models_dir = os.path.join(project_root, "models")
            os.makedirs(abs_models_dir, exist_ok=True)
            
            logger.info(f"Created necessary directories")
            logger.info(f"Models will be saved to: {abs_models_dir}")
        except Exception as e:
            logger.warning(f"Error creating directories: {e}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        print(f"Error: Data file not found: {data_path}")
        print("Please provide a valid data path using --data-path")
        return 1
    
    # Run the training process with improved parameters
    try:
        results = train_main(
            use_sample=args.use_sample,
            max_features=args.max_features,
            data_path=data_path,
            balance_strategy=args.balance
        )
        
        # Print summary of results
        print("\n" + "=" * 80)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 80)
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Validation AUC: {results['val_auc']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test AUC: {results['test_auc']:.4f}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        
        # Provide helpful suggestions based on common errors
        if "ratio required to generate new sample" in str(e):
            print("\nSuggestion: Try using a different balancing strategy:")
            print("  python train_improved_model.py --balance oversample")
            print("  python train_improved_model.py --balance none")
        elif "memory" in str(e).lower():
            print("\nSuggestion: Try reducing the feature count:")
            print("  python train_improved_model.py --max-features 5000")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 