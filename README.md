# PhishingNeuralGuardian-ML
Phishing emails detector with 97.36% accuracy using Multi-layer Perceptron neural networks, supervised classification, and sentiment analysis. Combines TF-IDF vectorization, URL pattern recognition, and threshold optimization in a production-ready pipeline, transforming raw emails into actionable security intelligence against cyber threats.


python src/evaluate_model.py --test-data samples/test_emails.csv --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --output-dir reports/evaluation

python src/evaluate_model.py --test-data samples/test_emails.csv --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --output-dir reports/evaluation

python src/test_model.py samples --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --threshold 0.01

Working 

python src/evaluate_model.py --test-data samples/test_emails.csv --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --output-dir reports/evaluation


python src/test_model.py samples/test_emails.csv --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --index 2 --threshold 0.01

python src/test_model.py samples/test_email.txt --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --threshold 0.01

python src/test_model.py samples --model models/mlp_sklearn.pkl --vectorizer models/tfidf.pkl --threshold 0.01




