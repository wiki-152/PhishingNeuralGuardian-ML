#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate a Jupyter notebook file for feature engineering demonstration.
"""

import json
import os
from pathlib import Path

def create_notebook():
    """Create a Jupyter notebook for feature engineering demonstration."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Feature Engineering for Phishing Email Detection\n",
                    "\n",
                    "This notebook demonstrates how to use the `features.py` module to create features for phishing email detection."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Add the src directory to the path\n",
                    "sys.path.append('../src')\n",
                    "from features import make_features, count_urls, count_urgency_words, detect_spoofing"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Load the dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the dataset\n",
                    "data_path = Path(\"../data/processed/all_combined.csv\")\n",
                    "df = pd.read_csv(data_path)\n",
                    "\n",
                    "# Display basic information\n",
                    "print(f\"Dataset size: {len(df)} emails\")\n",
                    "print(f\"Columns: {df.columns.tolist()}\")\n",
                    "print(\"\\nLabel distribution:\")\n",
                    "print(df['label'].value_counts())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Filter to only use labeled data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Filter to only use rows with known labels\n",
                    "labeled_df = df[df['label'].isin(['phishing', 'legitimate'])].reset_index(drop=True)\n",
                    "print(f\"Labeled dataset size: {len(labeled_df)} emails\")\n",
                    "print(\"\\nLabel distribution:\")\n",
                    "print(labeled_df['label'].value_counts())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Explore the numeric features"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Combine subject and body for text features\n",
                    "labeled_df['text'] = labeled_df['subject'].fillna('') + ' ' + labeled_df['body'].fillna('')\n",
                    "\n",
                    "# Calculate numeric features\n",
                    "labeled_df['word_count'] = labeled_df['text'].apply(lambda x: len(str(x).split()))\n",
                    "labeled_df['url_count'] = labeled_df['text'].apply(count_urls)\n",
                    "labeled_df['urgency_hits'] = labeled_df['text'].apply(count_urgency_words)\n",
                    "labeled_df['spoof_flag'] = labeled_df.apply(\n",
                    "    lambda row: detect_spoofing(row['from'], row.get('reply_to')), \n",
                    "    axis=1\n",
                    ")\n",
                    "\n",
                    "# Display summary statistics\n",
                    "print(\"Summary statistics for numeric features:\")\n",
                    "print(labeled_df[['word_count', 'url_count', 'urgency_hits', 'spoof_flag']].describe())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Compare features between phishing and legitimate emails"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Group by label and calculate mean\n",
                    "feature_comparison = labeled_df.groupby('label')[['word_count', 'url_count', 'urgency_hits', 'spoof_flag']].mean()\n",
                    "print(\"Average feature values by label:\")\n",
                    "print(feature_comparison)\n",
                    "\n",
                    "# Plot the comparison\n",
                    "feature_comparison.plot(kind='bar', figsize=(10, 6))\n",
                    "plt.title('Average Feature Values by Email Type')\n",
                    "plt.ylabel('Average Value')\n",
                    "plt.xticks(rotation=0)\n",
                    "plt.legend(title='Features')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Create features using the make_features function"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create features\n",
                    "output_dir = Path(\"../models/features\")\n",
                    "X, y, vectorizer, numeric_cols = make_features(labeled_df, max_tfidf=5000, output_dir=output_dir)\n",
                    "\n",
                    "print(f\"Feature matrix shape: {X.shape}\")\n",
                    "print(f\"Number of TF-IDF features: {X.shape[1] - len(numeric_cols)}\")\n",
                    "print(f\"Numeric features: {numeric_cols}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Explore the top TF-IDF features"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Get feature names\n",
                    "feature_names = vectorizer.get_feature_names_out().tolist()\n",
                    "\n",
                    "# Convert sparse matrix to array for the TF-IDF features only\n",
                    "X_tfidf = X.tocsr()[:, :len(feature_names)]\n",
                    "\n",
                    "# Calculate average TF-IDF values for phishing and legitimate emails\n",
                    "phishing_indices = np.where(y == 1)[0]\n",
                    "legitimate_indices = np.where(y == 0)[0]\n",
                    "\n",
                    "phishing_avg = X_tfidf[phishing_indices].mean(axis=0).A1\n",
                    "legitimate_avg = X_tfidf[legitimate_indices].mean(axis=0).A1\n",
                    "\n",
                    "# Find top features for phishing emails\n",
                    "top_phishing_indices = phishing_avg.argsort()[-20:][::-1]\n",
                    "top_phishing_features = [(feature_names[i], phishing_avg[i]) for i in top_phishing_indices]\n",
                    "\n",
                    "print(\"Top 20 TF-IDF features for phishing emails:\")\n",
                    "for feature, value in top_phishing_features:\n",
                    "    print(f\"{feature}: {value:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Save a sample of the data for model training"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save a smaller sample for quick model training\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "# Create a balanced sample\n",
                    "phishing_df = labeled_df[labeled_df['label'] == 'phishing'].sample(n=min(5000, len(labeled_df[labeled_df['label'] == 'phishing'])))\n",
                    "legitimate_df = labeled_df[labeled_df['label'] == 'legitimate'].sample(n=min(5000, len(labeled_df[labeled_df['label'] == 'legitimate'])))\n",
                    "sample_df = pd.concat([phishing_df, legitimate_df]).reset_index(drop=True)\n",
                    "\n",
                    "# Save the sample\n",
                    "sample_path = Path(\"../data/processed/sample_balanced.csv\")\n",
                    "sample_df.to_csv(sample_path, index=False)\n",
                    "\n",
                    "print(f\"Saved balanced sample with {len(sample_df)} emails to {sample_path}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Convert code cells from lists to strings
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "\n".join(cell["source"])
        elif cell["cell_type"] == "markdown":
            cell["source"] = "\n".join(cell["source"])
    
    # Create notebooks directory if it doesn't exist
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    
    # Write the notebook to a file
    notebook_path = notebooks_dir / "feature_engineering_demo.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook created at {notebook_path}")

if __name__ == "__main__":
    create_notebook() 