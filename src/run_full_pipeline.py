#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the entire phishing detection pipeline from data processing to model evaluation and report generation.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path
import pandas as pd

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

def run_command(command, description=None):
    """Run a command and log the output."""
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Command completed with exit code: {result.returncode}")
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"STDERR: {line}")
                
        return True, result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code: {e.returncode}")
        
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"STDERR: {line}")
                
        return False, e
        
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False, e

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if data directory exists
    data_dir = Path("../data")
    if not data_dir.exists():
        logger.error("Data directory not found. Please create it first.")
        return False
    
    # Check if processed data directory exists
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if models directory exists
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if reports directory exists
    reports_dir = Path("../reports")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if combined data file exists
    combined_file = processed_dir / "all_combined.csv"
    if not combined_file.exists():
        logger.warning(f"Combined data file not found: {combined_file}")
        logger.warning("You may need to run data processing steps first.")
    
    return True

def run_pipeline(args):
    """Run the full pipeline."""
    start_time = time.time()
    logger.info("Starting full phishing detection pipeline...")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Step 1: Create sample if needed
    if args.create_sample or args.force_all:
        success, _ = run_command(
            f"python create_sample.py --size {args.sample_size}",
            "Creating sample dataset"
        )
        if not success and not args.continue_on_error:
            return False
    
    # Step 2: Train the model
    if args.train or args.force_all:
        model_cmd = "python train_mlp_sklearn.py"
        # By default, use the full dataset (no --use-sample flag)
        if args.cpu_cores:
            model_cmd += f" --cpu-cores {args.cpu_cores}"
        
        success, _ = run_command(model_cmd, "Training the model")
        if not success and not args.continue_on_error:
            return False
    
    # Step 3: Evaluate the model on test data
    if args.evaluate or args.force_all:
        eval_cmd = f"python evaluate_model.py --test-data ../data/processed/test_sample.csv"
        if args.output_dir:
            eval_cmd += f" --output-dir {args.output_dir}"
        
        success, _ = run_command(eval_cmd, "Evaluating the model")
        if not success and not args.continue_on_error:
            return False
    
    # Step 4: Generate HTML report
    if args.generate_report or args.force_all:
        output_dir = args.output_dir or "../reports/evaluation"
        report_cmd = f"python generate_report.py --predictions {output_dir}/predictions.csv"
        if args.report_dir:
            report_cmd += f" --output-dir {args.report_dir}"
        
        success, _ = run_command(report_cmd, "Generating HTML report")
        if not success and not args.continue_on_error:
            return False
    
    # Calculate and log total time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Pipeline completed in {total_time:.2f} seconds")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PHISHING DETECTION PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds")
    
    if args.report_dir:
        report_path = Path(args.report_dir) / "report.html"
    else:
        report_path = Path(args.output_dir or "../reports/evaluation") / "report.html"
    
    if report_path.exists():
        print(f"HTML report generated: {report_path}")
    
    return True

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full phishing detection pipeline")
    
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force run all pipeline steps"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of emails to include in the sample dataset"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )
    
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use the sample dataset for training"
    )
    
    parser.add_argument(
        "--cpu-cores",
        type=int,
        help="Number of CPU cores to use for training"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model on test data"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default="samples/test_emails.csv",
        help="Path to test data CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--report-dir",
        type=str,
        help="Directory to save HTML report"
    )
    
    args = parser.parse_args()
    
    # If no specific steps are specified, run all steps
    if not any([args.create_sample, args.train, args.evaluate, args.generate_report]):
        args.force_all = True
    
    return args

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args) 