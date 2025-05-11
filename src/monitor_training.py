#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitor the training progress by checking the model directory and logs.
"""

import os
import time
import sys
from pathlib import Path
import psutil
import argparse

def print_with_timestamp(message):
    """Print a message with a timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def check_memory():
    """Check and print memory usage."""
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    used_percent = psutil.virtual_memory().percent
    
    print_with_timestamp(f"Memory: {available_memory_gb:.2f} GB available out of {total_memory_gb:.2f} GB total ({used_percent}% used)")

def check_cpu():
    """Check and print CPU usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    print_with_timestamp(f"CPU usage: {cpu_percent}%")

def check_models_dir():
    """Check the models directory for new files."""
    model_dir = Path("models")
    
    if not model_dir.exists():
        print_with_timestamp("Models directory does not exist.")
        return
    
    files = list(model_dir.glob("*"))
    
    if not files:
        print_with_timestamp("No files found in models directory.")
        return
    
    print_with_timestamp(f"Files in models directory ({len(files)}):")
    for file in files:
        if file.name == ".gitkeep":
            continue
            
        size_mb = os.path.getsize(file) / (1024 * 1024)
        modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file)))
        print(f"  - {file.name} ({size_mb:.2f} MB, modified: {modified_time})")

def check_log_file(log_file="pipeline.log", num_lines=10):
    """Check the last few lines of the log file."""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print_with_timestamp(f"Log file {log_file} does not exist.")
        return
    
    print_with_timestamp(f"Last {num_lines} lines from {log_file}:")
    
    with open(log_path, "r") as f:
        lines = f.readlines()
        
    for line in lines[-num_lines:]:
        print(f"  {line.strip()}")

def check_running_processes():
    """Check if any Python processes are running."""
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not python_processes:
        print_with_timestamp("No Python processes running.")
        return
    
    print_with_timestamp(f"Running Python processes ({len(python_processes)}):")
    for proc in python_processes:
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else "N/A"
            print(f"  - PID {proc.info['pid']}: {cmdline} (CPU: {proc.info['cpu_percent']}%, MEM: {proc.info['memory_percent']:.1f}%)")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def monitor(interval=5, log_file="pipeline.log", num_lines=10):
    """Monitor the training progress."""
    try:
        while True:
            print("\n" + "=" * 80)
            print_with_timestamp("Monitoring training progress")
            print("=" * 80)
            
            check_memory()
            check_cpu()
            print()
            
            check_running_processes()
            print()
            
            check_models_dir()
            print()
            
            check_log_file(log_file, num_lines)
            
            print("\nPress Ctrl+C to stop monitoring...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Monitor training progress")
    
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="pipeline.log",
        help="Log file to monitor"
    )
    
    parser.add_argument(
        "--num-lines",
        type=int,
        default=10,
        help="Number of log lines to show"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    monitor(args.interval, args.log_file, args.num_lines) 