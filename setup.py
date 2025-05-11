from setuptools import setup, find_packages

setup(
    name="phishing_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=0.24.0",
        "pandas>=1.1.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
        "nltk>=3.5",
        "imbalanced-learn>=0.8.0",
        "joblib>=1.0.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        "console_scripts": [
            "phishing-detector=phishing_detector.main:main",
        ],
    },
)
