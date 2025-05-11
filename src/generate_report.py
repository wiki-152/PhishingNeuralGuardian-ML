#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a comprehensive HTML report for the phishing email detection system.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Email Detection System Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .section {
            margin-bottom: 40px;
            border: 1px solid #eee;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .example {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .img-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            border-top: 1px solid #eee;
            padding-top: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .phishing {
            color: #e74c3c;
        }
        .legitimate {
            color: #27ae60;
        }
        .incorrect {
            background-color: #ffecec;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Phishing Email Detection System Report</h1>
        <p>Generated on {{ date }}</p>
    </div>
    
    <div class="section">
        <h2>1. Model Performance Overview</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="metric-value">{{ metrics.accuracy|round(4) }}</div>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="metric-value">{{ metrics.precision|round(4) }}</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="metric-value">{{ metrics.recall|round(4) }}</div>
            </div>
            <div class="metric-card">
                <h3>F1 Score</h3>
                <div class="metric-value">{{ metrics.f1|round(4) }}</div>
            </div>
            <div class="metric-card">
                <h3>AUC</h3>
                <div class="metric-value">{{ metrics.auc|round(4) }}</div>
            </div>
        </div>
        
        <div class="img-container">
            <h3>Confusion Matrix</h3>
            <img src="{{ cm_path }}" alt="Confusion Matrix">
        </div>
    </div>
    
    <div class="section">
        <h2>2. Prediction Results</h2>
        <h3>Summary</h3>
        <p>Tested on {{ total_emails }} emails: {{ phishing_count }} phishing and {{ legitimate_count }} legitimate.</p>
        <p>Correctly classified: {{ correct_count }} ({{ correct_pct }}%)</p>
        <p>Misclassified: {{ incorrect_count }} ({{ incorrect_pct }}%)</p>
        
        <h3>Predictions Table</h3>
        <table>
            <tr>
                <th>Subject</th>
                <th>Actual</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Result</th>
            </tr>
            {% for row in predictions %}
            <tr {% if not row.correct %}class="incorrect"{% endif %}>
                <td>{{ row.subject }}</td>
                <td class="{% if row.label == 'phishing' %}phishing{% else %}legitimate{% endif %}">
                    {{ row.label }}
                </td>
                <td class="{% if row.prediction_label == 'phishing' %}phishing{% else %}legitimate{% endif %}">
                    {{ row.prediction_label }}
                </td>
                <td>{{ row.phishing_probability|round(4) }}</td>
                <td>{{ "✓" if row.correct else "✗" }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    {% if misclassified_count > 0 %}
    <div class="section">
        <h2>3. Misclassified Examples</h2>
        {% for row in misclassified %}
        <div class="example {% if not row.correct %}incorrect{% endif %}">
            <h3>Example {{ loop.index }}</h3>
            <p><strong>Subject:</strong> {{ row.subject }}</p>
            <p><strong>Body:</strong> {{ row.body }}</p>
            <p><strong>True label:</strong> <span class="{% if row.label == 'phishing' %}phishing{% else %}legitimate{% endif %}">{{ row.label }}</span></p>
            <p><strong>Predicted:</strong> <span class="{% if row.prediction_label == 'phishing' %}phishing{% else %}legitimate{% endif %}">{{ row.prediction_label }}</span> (probability: {{ row.phishing_probability|round(4) }})</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>4. Recommendations</h2>
        <p>Based on the evaluation results, here are some recommendations to improve the model:</p>
        <ul>
            <li>The model has a tendency to classify legitimate emails as phishing, especially those with links or keywords commonly found in phishing emails. Consider improving feature engineering to better distinguish between legitimate and phishing links.</li>
            <li>Add more contextual features to help the model better understand the semantics of the emails.</li>
            <li>Increase the training dataset with more diverse examples of legitimate emails containing links.</li>
            <li>Consider fine-tuning the model threshold to reduce false positives if preserving legitimate emails is more important than catching all phishing attempts.</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Phishing Email Detection System © 2025</p>
    </div>
</body>
</html>
"""

def generate_confusion_matrix_image(predictions_df, output_dir):
    """Generate and save confusion matrix image."""
    # Create confusion matrix
    y_true = predictions_df['true_label']
    y_pred = predictions_df['prediction']
    
    labels = ['Legitimate', 'Phishing']
    cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], normalize=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    return cm_path

def generate_report(predictions_path, metrics_path=None, output_dir=None):
    """
    Generate an HTML report from evaluation results.
    
    Args:
        predictions_path: Path to predictions CSV file
        metrics_path: Path to metrics CSV file (optional)
        output_dir: Directory to save the report (default is same as predictions_path)
    """
    # Set output directory
    predictions_path = Path(predictions_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = predictions_path.parent
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    
    # Load metrics if available
    metrics = {}
    if metrics_path:
        logger.info(f"Loading metrics from {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.iloc[0].to_dict()
    else:
        # Calculate metrics from predictions
        logger.info("Calculating metrics from predictions")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        if 'true_label' in predictions_df.columns and 'prediction' in predictions_df.columns:
            y_true = predictions_df['true_label']
            y_pred = predictions_df['prediction']
            y_prob = predictions_df['phishing_probability']
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    # Generate confusion matrix image
    cm_path = generate_confusion_matrix_image(predictions_df, output_dir)
    
    # Calculate additional stats
    total_emails = len(predictions_df)
    phishing_count = sum(predictions_df['true_label'] == 1)
    legitimate_count = sum(predictions_df['true_label'] == 0)
    
    if 'correct' not in predictions_df.columns:
        predictions_df['correct'] = predictions_df['prediction'] == predictions_df['true_label']
    
    correct_count = sum(predictions_df['correct'])
    incorrect_count = total_emails - correct_count
    correct_pct = (correct_count / total_emails) * 100
    incorrect_pct = (incorrect_count / total_emails) * 100
    
    # Get misclassified examples
    misclassified = predictions_df[~predictions_df['correct']].to_dict('records')
    misclassified_count = len(misclassified)
    
    # Prepare template data
    template_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'cm_path': str(cm_path.relative_to(output_dir)),
        'total_emails': total_emails,
        'phishing_count': phishing_count,
        'legitimate_count': legitimate_count,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'correct_pct': round(correct_pct, 2),
        'incorrect_pct': round(incorrect_pct, 2),
        'predictions': predictions_df.to_dict('records'),
        'misclassified': misclassified,
        'misclassified_count': misclassified_count
    }
    
    # Render template
    template = Template(HTML_TEMPLATE)
    html_content = template.render(**template_data)
    
    # Save report
    report_path = output_dir / 'report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report generated and saved to {report_path}")
    return report_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate HTML report for phishing detection system")
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV file"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics CSV file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the report (default is same as predictions)"
    )
    
    args = parser.parse_args()
    
    # Generate report
    report_path = generate_report(args.predictions, args.metrics, args.output_dir)
    
    print(f"Report generated successfully: {report_path}")

if __name__ == "__main__":
    main() 