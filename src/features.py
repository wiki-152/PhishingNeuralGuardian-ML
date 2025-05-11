#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering for phishing email detection.
Combines TF-IDF features with numeric features.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
from pathlib import Path
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import logging
from tqdm import tqdm
import traceback
import urllib.parse
import socket
import ssl
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
    logger.debug("VADER lexicon already downloaded")
except LookupError:
    logger.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
    logger.debug("Punkt tokenizer already downloaded")
except LookupError:
    logger.info("Downloading Punkt tokenizer...")
    nltk.download('punkt')

# Additional resources needed for text complexity calculation
try:
    nltk.data.find('tokenizers/punkt/english.pickle')
    logger.debug("English punkt tokenizer already downloaded")
except LookupError:
    logger.info("Downloading English punkt tokenizer...")
    nltk.download('punkt')

# Common legitimate domains (for link analysis)
COMMON_LEGITIMATE_DOMAINS = {
    'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 
    'twitter.com', 'linkedin.com', 'github.com', 'dropbox.com', 'zoom.us',
    'office.com', 'live.com', 'outlook.com', 'gmail.com', 'yahoo.com',
    'instagram.com', 'youtube.com', 'netflix.com', 'spotify.com', 'slack.com',
    'teams.microsoft.com', 'drive.google.com', 'docs.google.com', 'sharepoint.com'
}

def sentiment_features(text):
    """
    Extract sentiment features from text using VADER.
    
    Args:
        text: String, the text to analyze
        
    Returns:
        Dictionary with sentiment scores: compound, neg, neu, pos
    """
    try:
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(str(text))
        
        return {
            'compound': sentiment_scores['compound'],  # Range: -1 (negative) to 1 (positive)
            'neg': sentiment_scores['neg'],            # Range: 0 to 1
            'neu': sentiment_scores['neu'],            # Range: 0 to 1
            'pos': sentiment_scores['pos']             # Range: 0 to 1
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        logger.debug(traceback.format_exc())
        return {
            'compound': 0.0,
            'neg': 0.0,
            'neu': 1.0,
            'pos': 0.0
        }

def extract_urls(text):
    """Extract all URLs from text."""
    try:
        # Regex to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls = url_pattern.findall(str(text))
        return urls
    except Exception as e:
        logger.error(f"Error extracting URLs: {e}")
        return []

def count_urls(text):
    """Count the number of URLs in the text."""
    return len(extract_urls(text))

def analyze_url(url):
    """
    Analyze URL for phishing indicators.
    
    Returns:
        Dictionary with URL features
    """
    features = {
        'url_length': 0,
        'num_dots': 0,
        'num_hyphens': 0,
        'num_at_signs': 0,
        'num_digits': 0,
        'num_subdomains': 0,
        'has_ip_address': 0,
        'is_shortened': 0,
        'has_suspicious_tld': 0,
        'is_common_domain': 0,
        'has_https': 0
    }
    
    try:
        # Clean URL
        url = url.strip()
        
        # Basic URL features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_at_signs'] = url.count('@')
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Check for HTTPS
        features['has_https'] = 1 if url.startswith('https://') else 0
        
        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url if url.startswith(('http://', 'https://')) else f"http://{url}")
            netloc = parsed.netloc
            
            # Check for IP address
            features['has_ip_address'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', netloc) else 0
            
            # Count subdomains
            if netloc:
                subdomains = netloc.split('.')
                features['num_subdomains'] = len(subdomains) - 1 if len(subdomains) > 1 else 0
                
                # Check for common legitimate domains
                domain = '.'.join(subdomains[-2:]) if len(subdomains) > 1 else netloc
                features['is_common_domain'] = 1 if domain.lower() in COMMON_LEGITIMATE_DOMAINS else 0
                
                # Check for suspicious TLDs
                tld = subdomains[-1].lower() if subdomains else ''
                suspicious_tlds = ['xyz', 'top', 'tk', 'gq', 'ml', 'ga', 'cf', 'ru', 'cn']
                features['has_suspicious_tld'] = 1 if tld in suspicious_tlds else 0
        except Exception:
            pass
        
        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'j.mp']
        features['is_shortened'] = 1 if any(shortener in url.lower() for shortener in shorteners) else 0
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
    
    return features

def analyze_urls_in_text(text):
    """
    Analyze all URLs in text and return aggregated features.
    
    Args:
        text: Text containing URLs
        
    Returns:
        Dictionary with aggregated URL features
    """
    urls = extract_urls(text)
    
    if not urls:
        return {
            'url_count': 0,
            'avg_url_length': 0,
            'max_url_length': 0,
            'avg_num_dots': 0,
            'max_num_dots': 0,
            'avg_num_hyphens': 0,
            'max_num_hyphens': 0,
            'has_ip_address': 0,
            'has_shortened_url': 0,
            'has_suspicious_tld': 0,
            'has_common_domain': 0,
            'https_ratio': 0
        }
    
    # Analyze each URL
    url_features = [analyze_url(url) for url in urls]
    
    # Aggregate features
    aggregated = {
        'url_count': len(urls),
        'avg_url_length': np.mean([f['url_length'] for f in url_features]),
        'max_url_length': max([f['url_length'] for f in url_features]),
        'avg_num_dots': np.mean([f['num_dots'] for f in url_features]),
        'max_num_dots': max([f['num_dots'] for f in url_features]),
        'avg_num_hyphens': np.mean([f['num_hyphens'] for f in url_features]),
        'max_num_hyphens': max([f['num_hyphens'] for f in url_features]),
        'has_ip_address': any(f['has_ip_address'] for f in url_features),
        'has_shortened_url': any(f['is_shortened'] for f in url_features),
        'has_suspicious_tld': any(f['has_suspicious_tld'] for f in url_features),
        'has_common_domain': any(f['is_common_domain'] for f in url_features),
        'https_ratio': sum(f['has_https'] for f in url_features) / len(urls)
    }
    
    # Convert boolean values to int
    for key in ['has_ip_address', 'has_shortened_url', 'has_suspicious_tld', 'has_common_domain']:
        aggregated[key] = int(aggregated[key])
    
    return aggregated

def count_urgency_words(text):
    """Count words that indicate urgency."""
    try:
        # List of words indicating urgency
        urgency_words = [
            'urgent', 'immediately', 'attention', 'important', 'action', 
            'required', 'verify', 'confirm', 'update', 'security', 'alert',
            'warning', 'limited', 'expires', 'deadline', 'asap', 'now',
            'quick', 'critical', 'urgent', 'validate', 'suspend', 'restricted',
            'unauthorized', 'blocked', 'locked', 'disabled', 'compromised'
        ]
        
        # Convert to lowercase and count matches
        text = str(text).lower()
        count = sum(1 for word in urgency_words if word in text)
        logger.debug(f"Found {count} urgency words in text")
        return count  # Typically, 3+ hits indicates high urgency content
    except Exception as e:
        logger.error(f"Error counting urgency words: {e}")
        logger.debug(traceback.format_exc())
        return 0

def count_financial_words(text):
    """Count words related to financial topics."""
    try:
        financial_words = [
            'bank', 'account', 'credit', 'debit', 'card', 'payment', 'transaction',
            'transfer', 'money', 'balance', 'deposit', 'withdraw', 'financial',
            'paypal', 'bitcoin', 'crypto', 'wallet', 'invoice', 'bill', 'tax',
            'refund', 'cash', 'statement', 'loan', 'mortgage', 'interest', 'fee'
        ]
        
        text = str(text).lower()
        count = sum(1 for word in financial_words if word in text)
        return count
    except Exception as e:
        logger.error(f"Error counting financial words: {e}")
        return 0

def count_personal_info_words(text):
    """Count words related to personal information."""
    try:
        personal_info_words = [
            'password', 'username', 'login', 'credentials', 'ssn', 'social security',
            'identity', 'id', 'driver license', 'passport', 'date of birth', 'address',
            'phone', 'email', 'verification', 'verify', 'confirm', 'update', 'personal',
            'information', 'details', 'data', 'profile'
        ]
        
        text = str(text).lower()
        count = sum(1 for word in personal_info_words if word in text)
        return count
    except Exception as e:
        logger.error(f"Error counting personal info words: {e}")
        return 0

def calculate_text_complexity(text):
    """Calculate text complexity features."""
    try:
        text = str(text)
        
        # Basic text complexity metrics that don't require tokenization
        if not text:
            return {
                'avg_word_length': 0,
                'unique_words_ratio': 0,
                'capital_letters_ratio': 0
            }
            
        # Simple word splitting (fallback if tokenization fails)
        words = text.split()
        
        # Calculate metrics using simple word splitting
        if words:
            avg_word_length = sum(len(word) for word in words if word.isalpha()) / max(1, sum(word.isalpha() for word in words))
            unique_words_ratio = len(set(words)) / len(words) if words else 0
        else:
            avg_word_length = 0
            unique_words_ratio = 0
            
        # Capital letters ratio (doesn't require tokenization)
        capital_letters = sum(1 for c in text if c.isupper())
        capital_letters_ratio = capital_letters / len(text) if text else 0
        
        # Try to use NLTK tokenization if available
        try:
            tokens = word_tokenize(text)
            
            # Refine metrics with proper tokenization if available
            if tokens:
                avg_word_length = np.mean([len(word) for word in tokens if word.isalpha()])
                unique_words_ratio = len(set(tokens)) / len(tokens)
        except Exception as inner_e:
            # If tokenization fails, use the simple metrics calculated earlier
            logger.debug(f"Using fallback text complexity metrics: {inner_e}")
        
        return {
            'avg_word_length': avg_word_length if not np.isnan(avg_word_length) else 0,
            'unique_words_ratio': unique_words_ratio,
            'capital_letters_ratio': capital_letters_ratio
        }
    except Exception as e:
        logger.error(f"Error calculating text complexity: {e}")
        return {
            'avg_word_length': 0,
            'unique_words_ratio': 0,
            'capital_letters_ratio': 0
        }

def detect_spoofing(from_field, reply_to_field=None):
    """
    Detect potential email spoofing by checking for mismatches between
    'from' and 'reply-to' fields, or suspicious patterns in the 'from' field.
    """
    try:
        # Convert inputs to strings and lowercase
        from_field = str(from_field).lower()
        reply_to_field = str(reply_to_field).lower() if reply_to_field else ""
        
        # Check for mismatch between from and reply-to domains
        from_domain = re.search(r'@([^>@\s]+)', from_field)
        reply_domain = re.search(r'@([^>@\s]+)', reply_to_field)
        
        # If both domains exist and don't match, potential spoofing
        if from_domain and reply_domain and from_domain.group(1) != reply_domain.group(1):
            logger.debug(f"Spoofing detected: from domain {from_domain.group(1)} != reply-to domain {reply_domain.group(1)}")
            return 1
        
        # Check for suspicious patterns in from field
        suspicious_patterns = [
            r'@.*@',                  # Multiple @ symbols
            r'[^a-zA-Z0-9.@_-]',      # Special characters in email
            r'\.(ru|cn|tk|top|xyz)$'  # Suspicious TLDs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, from_field):
                logger.debug(f"Suspicious pattern detected in from field: {pattern}")
                return 1
                
        return 0
    except Exception as e:
        logger.error(f"Error in spoofing detection: {e}")
        logger.debug(traceback.format_exc())
        return 0

def extract_context_features(text):
    """
    Extract contextual features from text.
    
    Args:
        text: String, the text to analyze
        
    Returns:
        Dictionary with contextual features
    """
    try:
        text = str(text).lower()
        
        # Check for greeting patterns
        has_greeting = bool(re.search(r'\b(hi|hello|dear|greetings|good morning|good afternoon|good evening)\b', text))
        
        # Check for signature patterns
        has_signature = bool(re.search(r'\b(regards|sincerely|thank you|thanks|best|cheers|yours truly|yours sincerely)\b', text))
        
        # Check for formal business language
        business_terms = [
            'meeting', 'conference', 'project', 'report', 'team', 'client',
            'deadline', 'schedule', 'agenda', 'presentation', 'document',
            'review', 'approve', 'discuss', 'follow up', 'update'
        ]
        business_term_count = sum(1 for term in business_terms if term in text)
        
        # Check for request patterns
        has_request = bool(re.search(r'\b(please|kindly|would you|could you|can you)\b', text))
        
        # Check for question patterns
        has_question = bool(re.search(r'\?', text))
        
        # Check for personal communication patterns
        personal_terms = [
            'we', 'our', 'us', 'team', 'I', 'my', 'me', 'mine',
            'colleague', 'coworker', 'department', 'office'
        ]
        personal_term_count = sum(1 for term in personal_terms if re.search(fr'\b{term}\b', text))
        
        return {
            'has_greeting': int(has_greeting),
            'has_signature': int(has_signature),
            'business_term_count': business_term_count,
            'has_request': int(has_request),
            'has_question': int(has_question),
            'personal_term_count': personal_term_count
        }
    except Exception as e:
        logger.error(f"Error extracting context features: {e}")
        return {
            'has_greeting': 0,
            'has_signature': 0,
            'business_term_count': 0,
            'has_request': 0,
            'has_question': 0,
            'personal_term_count': 0
        }

def make_features(df, max_tfidf=5000, output_dir=None):
    """
    Create features for phishing email detection.
    
    Args:
        df: DataFrame with 'subject', 'body', 'from', 'reply_to' columns
        max_tfidf: Maximum number of TF-IDF features to use
        output_dir: Directory to save the vectorizer and scaler
        
    Returns:
        X_sparse: Sparse matrix of features
        y: Target labels
        vectorizer: Fitted TF-IDF vectorizer
        numeric_cols: List of numeric column names
    """
    logger.info(f"Creating features for {len(df)} emails")
    
    # Ensure output directory exists
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output directory set to {output_dir}")
    
    try:
        # Check for required columns and add them if missing
        required_columns = ['subject', 'body', 'from', 'reply_to']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Adding empty column.")
                df[col] = ""
        
        # Combine subject and body for text features
        logger.info("Combining subject and body for text features")
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        
        # Create TF-IDF features
        logger.info(f"Creating TF-IDF features (max_features={max_tfidf})")
        vectorizer = TfidfVectorizer(
            max_features=max_tfidf,
            ngram_range=(1, 3),       # Use 1-3 grams
            min_df=5,                 # Ignore terms that appear in less than 5 documents
            max_df=0.8,               # Ignore terms that appear in more than 80% of documents
            stop_words='english',
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True         # Apply sublinear tf scaling (1 + log(tf))
        )
        
        # Fit and transform the text data
        logger.info("Fitting and transforming text data with TF-IDF")
        X_tfidf = vectorizer.fit_transform(df['text'])
        logger.info(f"TF-IDF shape: {X_tfidf.shape}")
        
        # Create numeric features
        logger.info("Creating numeric features")
        
        # Use tqdm for progress bar
        tqdm.pandas(desc="Calculating word counts")
        df['word_count'] = df['text'].progress_apply(lambda x: len(str(x).split()))
        
        # URL analysis
        logger.info("Analyzing URLs")
        tqdm.pandas(desc="Analyzing URLs")
        url_features = df['text'].progress_apply(analyze_urls_in_text)
        
        # Extract URL features
        for feature in ['url_count', 'avg_url_length', 'max_url_length', 'avg_num_dots', 
                       'max_num_dots', 'avg_num_hyphens', 'max_num_hyphens', 'has_ip_address',
                       'has_shortened_url', 'has_suspicious_tld', 'has_common_domain', 'https_ratio']:
            df[feature] = url_features.apply(lambda x: x[feature])
        
        # Sentiment analysis
        logger.info("Performing sentiment analysis")
        tqdm.pandas(desc="Analyzing sentiment")
        sentiments = df['text'].progress_apply(sentiment_features)
        df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
        df['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
        df['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
        df['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
        
        # Word pattern detection
        tqdm.pandas(desc="Detecting urgency words")
        df['urgency_count'] = df['text'].progress_apply(count_urgency_words)
        
        tqdm.pandas(desc="Detecting financial words")
        df['financial_count'] = df['text'].progress_apply(count_financial_words)
        
        tqdm.pandas(desc="Detecting personal info words")
        df['personal_info_count'] = df['text'].progress_apply(count_personal_info_words)
        
        # Text complexity features
        logger.info("Calculating text complexity")
        tqdm.pandas(desc="Calculating text complexity")
        complexity_features = df['text'].progress_apply(calculate_text_complexity)
        df['avg_word_length'] = complexity_features.apply(lambda x: x['avg_word_length'])
        df['unique_words_ratio'] = complexity_features.apply(lambda x: x['unique_words_ratio'])
        df['capital_letters_ratio'] = complexity_features.apply(lambda x: x['capital_letters_ratio'])
        
        # Context features
        logger.info("Extracting context features")
        tqdm.pandas(desc="Extracting context features")
        context_features = df['text'].progress_apply(extract_context_features)
        df['has_greeting'] = context_features.apply(lambda x: x['has_greeting'])
        df['has_signature'] = context_features.apply(lambda x: x['has_signature'])
        df['business_term_count'] = context_features.apply(lambda x: x['business_term_count'])
        df['has_request'] = context_features.apply(lambda x: x['has_request'])
        df['has_question'] = context_features.apply(lambda x: x['has_question'])
        df['personal_term_count'] = context_features.apply(lambda x: x['personal_term_count'])
        
        # Email spoofing detection
        tqdm.pandas(desc="Detecting spoofing")
        df['spoofing'] = df.progress_apply(
            lambda row: detect_spoofing(row['from'], row['reply_to']), axis=1
        )
        
        # Create list of all numeric feature columns
        numeric_cols = [
            'word_count', 
            'url_count', 'avg_url_length', 'max_url_length', 'avg_num_dots', 
            'max_num_dots', 'avg_num_hyphens', 'max_num_hyphens', 'has_ip_address',
            'has_shortened_url', 'has_suspicious_tld', 'has_common_domain', 'https_ratio',
            'sentiment_compound', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos',
            'urgency_count', 'financial_count', 'personal_info_count',
            'avg_word_length', 'unique_words_ratio', 'capital_letters_ratio',
            'has_greeting', 'has_signature', 'business_term_count',
            'has_request', 'has_question', 'personal_term_count',
            'spoofing'
        ]
        
        # Scale numeric features
        logger.info("Scaling numeric features")
        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(df[numeric_cols].fillna(0))
        
        # Combine TF-IDF and numeric features
        logger.info("Combining TF-IDF and numeric features")
        X_sparse = hstack([X_tfidf, numeric_features])
        
        # Prepare target variable
        logger.info("Preparing target variable")
        y = df['label'].map({'phishing': 1, 'legitimate': 0}).values
        
        # Save vectorizer and scaler if output directory is provided
        if output_dir:
            logger.info(f"Saving vectorizer and scaler to {output_dir}")
            joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
            joblib.dump(scaler, output_dir / "feature_scaler.pkl")
            
            # Save feature names for interpretability
            with open(output_dir / "feature_names.txt", "w") as f:
                f.write("TF-IDF features: " + ", ".join(vectorizer.get_feature_names_out()) + "\n")
                f.write("Numeric features: " + ", ".join(numeric_cols) + "\n")
        
        logger.info(f"Feature engineering complete. X shape: {X_sparse.shape}, y shape: {y.shape}")
        return X_sparse, y, vectorizer, numeric_cols
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    logger.info("Starting feature engineering example")
    
    try:
        # Load dataset
        data_path = Path("data/processed/all_combined.csv")
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} emails")
        
        # Filter to only use rows with known labels
        df = df[df['label'].isin(['phishing', 'legitimate'])].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} emails with known labels")
        
        # Create features
        output_dir = Path("models/features")
        logger.info(f"Creating features and saving to {output_dir}")
        X, y, vectorizer, numeric_cols = make_features(df, output_dir=output_dir)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of TF-IDF features: {X.shape[1] - len(numeric_cols)}")
        print(f"Numeric features: {numeric_cols}")
        print(f"Vectorizer and scaler saved to {output_dir}")
        
        logger.info("Feature engineering example completed successfully")
    
    except Exception as e:
        logger.error(f"Error in feature engineering example: {e}")
        logger.debug(traceback.format_exc()) 