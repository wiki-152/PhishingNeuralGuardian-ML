#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the entire pipeline from data processing to model training.
"""

import os
import sys
import time
import logging
import traceback
import gc
from pathlib import Path
import argparse
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_memory():
    """Check and log available memory."""
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    used_percent = psutil.virtual_memory().percent
    
    logger.info(f"Memory: {available_memory_gb:.2f} GB available out of {total_memory_gb:.2f} GB total ({used_percent}% used)")
    
    return available_memory_gb, total_memory_gb, used_percent

def run_pipeline(args):
    """
    Run the entire pipeline from data processing to model training.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    logger.info("Starting pipeline")
    
    # Check memory
    available_memory_gb, _, _ = check_memory()
    
    # Create directories
    data_dir = Path("data/processed")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Create sample dataset if needed
    if args.create_sample or not (data_dir / "test_sample.csv").exists():
        logger.info("Step 1: Creating sample dataset")
        
        # Import here to avoid loading unnecessary modules
        from create_sample import create_sample
        
        input_path = data_dir / "all_combined.csv"
        output_path = data_dir / "test_sample.csv"
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            logger.info("Please run the data processing scripts first to create all_combined.csv")
            return
        
        # Calculate appropriate sample size based on available memory
        file_size_gb = os.path.getsize(input_path) / (1024 * 1024 * 1024)
        logger.info(f"Input file size: {file_size_gb:.2f} GB")
        
        # If file is large and memory is limited, use a smaller sample
        if file_size_gb > 1.0 and available_memory_gb < 8.0:
            # Estimate a reasonable sample size based on available memory
            # Assume we need about 5x the file size for processing
            sample_size = int((available_memory_gb * 0.2) / file_size_gb * 100000)
            sample_size = max(1000, min(sample_size, 50000))  # Between 1,000 and 50,000
            logger.info(f"Using a reduced sample size of {sample_size} due to memory constraints")
        else:
            sample_size = args.sample_size
            logger.info(f"Using requested sample size of {sample_size}")
        
        try:
            create_sample(input_path, output_path, sample_size=sample_size, balanced=True)
            
            # Force garbage collection
            gc.collect()
            check_memory()
            
        except Exception as e:
            logger.error(f"Error creating sample: {e}")
            logger.debug(traceback.format_exc())
            return
    else:
        logger.info("Step 1: Skipping sample creation (already exists)")
    
    # Step 2: Train the model
    logger.info("Step 2: Training the model")
    
    try:
        if args.model_type == "sklearn":
            logger.info("Using scikit-learn MLP model")
            # Import here to avoid loading unnecessary modules
            from train_mlp_sklearn import main as train_sklearn
            
            # Determine whether to use sample or full dataset
            use_sample = args.use_sample
            
            # If memory is limited, force using sample
            if available_memory_gb < 4.0 and not use_sample:
                logger.warning("Limited memory detected. Forcing use of sample dataset.")
                use_sample = True
            
            # Determine max features based on available memory
            if available_memory_gb < 4.0:
                max_features = 2000
            elif available_memory_gb < 8.0:
                max_features = 3000
            else:
                max_features = 5000
                
            logger.info(f"Using {'sample' if use_sample else 'full'} dataset with {max_features} max features")
            
            # Train the model
            results = train_sklearn(use_sample=use_sample, max_features=max_features)
            
            if results:
                logger.info(f"Training results: Validation Accuracy: {results['val_accuracy']:.4f}, Test Accuracy: {results['test_accuracy']:.4f}")
            
        else:
            logger.info("Using PyTorch MLP model")
            # Import here to avoid loading unnecessary modules
            from train_mlp import main as train_pytorch
            train_pytorch()
            
        # Force garbage collection
        gc.collect()
        check_memory()
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.debug(traceback.format_exc())
        return
    
    end_time = time.time()
    logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the phishing detection pipeline")
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset even if one already exists"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of emails to include in the sample dataset"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["sklearn", "pytorch"],
        default="sklearn",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use the sample dataset for training instead of the full dataset"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        run_pipeline(args)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc()) 