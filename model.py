import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

class ExpenseClassifier:
    def __init__(self):
        # TF-IDF extractor using character n-grams (2 to 5 letters) as specified in project design
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=1)
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Clean UPI handles, reference numbers, and special syntax
        text = re.sub(r'(upi|pos|neft|imps)/[0-9]+/', '', text, flags=re.IGNORECASE)
        text = re.sub(r'/[a-z0-9]+$', '', text, flags=re.IGNORECASE)
        return text.lower().strip()

    def train(self, df):
        """Train the classifier on historical/rule-backed data."""
        if df.empty or 'category' not in df.columns:
            return
        
        valid_df = df[df['category'] != 'Unassigned'].copy()
        if len(valid_df) < 2:
            return # Not enough classes to train

        X_raw = valid_df['description'].apply(self.clean_text)
        y = valid_df['category']

        X = self.vectorizer.fit_transform(X_raw)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_batch(self, df):
        """Predict categories and confidence scores for transactions."""
        if not self.is_trained or df.empty:
            df['category'] = 'Unassigned'
            df['confidence'] = 0.0
            return df

        X_raw = df['description'].apply(self.clean_text)
        X = self.vectorizer.transform(X_raw)
        
        preds = self.model.predict(X)
        probas = np.max(self.model.predict_proba(X), axis=1)

        df['category'] = preds
        df['confidence'] = probas
        # Mark as needing review if confidence < 60%
        df['is_reviewed'] = df['confidence'].apply(lambda c: 0 if c >= 0.6 else 1)
        return df