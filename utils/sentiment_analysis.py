"""
utils/sentiment_analysis.py
----------------------------
Financial news sentiment scoring using TF-IDF + Naive Bayes.
Designed for use with the FNSPID dataset or any CSV with
[headline, label] columns.
"""

from __future__ import annotations
import os, logging
import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

log = logging.getLogger(__name__)

# Download NLTK resources once
for resource in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Lowercase, tokenise, remove stopwords, and stem."""
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [
        _stemmer.stem(t)
        for t in tokens
        if t.isalpha() and t not in _stopwords
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Classifier
# ─────────────────────────────────────────────────────────────────────────────

class FinancialSentimentClassifier:
    """
    TF-IDF + Multinomial Naive Bayes pipeline for financial headline sentiment.
    Labels: positive / negative / neutral
    """

    def __init__(self, max_features: int = config.TFIDF_MAX_FEATURES):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features   = max_features,
                ngram_range    = (1, 2),
                sublinear_tf   = True,
                preprocessor   = preprocess_text,
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        self._fitted = False

    def fit(self, headlines: list[str], labels: list[str]) -> "FinancialSentimentClassifier":
        log.info("Fitting sentiment classifier on %d samples …", len(headlines))
        self.pipeline.fit(headlines, labels)
        self._fitted = True
        return self

    def predict(self, headlines: list[str]) -> np.ndarray:
        return self.pipeline.predict(headlines)

    def predict_proba(self, headlines: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(headlines)

    def score_headline(self, headline: str) -> dict:
        """Return sentiment label and class probabilities for a single headline."""
        label = self.predict([headline])[0]
        proba = self.predict_proba([headline])[0]
        classes = self.pipeline.classes_
        return {"label": label, "probabilities": dict(zip(classes, proba))}

    def evaluate(self, headlines: list[str], labels: list[str]) -> str:
        preds = self.predict(headlines)
        return classification_report(labels, preds)

    def save(self, path: str | None = None):
        path = path or os.path.join(config.MODELS_DIR, "sentiment_classifier.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info("Sentiment model saved → %s", path)

    @classmethod
    def load(cls, path: str | None = None) -> "FinancialSentimentClassifier":
        path = path or os.path.join(config.MODELS_DIR, "sentiment_classifier.pkl")
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader for FNSPID-style CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_fnspid(csv_path: str) -> pd.DataFrame:
    """
    Load the FNSPID dataset (or any CSV with 'headline' and 'sentiment' columns).
    Expected columns: Date, Ticker, headline, sentiment
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"headline", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    df = df.dropna(subset=["headline", "sentiment"])
    log.info("Loaded %d news records from %s", len(df), csv_path)
    return df


def train_from_csv(csv_path: str) -> FinancialSentimentClassifier:
    """Convenience: load FNSPID CSV, train and return the fitted classifier."""
    df = load_fnspid(csv_path)
    headlines = df["headline"].tolist()
    labels    = df["sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        headlines, labels, test_size=0.2, random_state=config.RANDOM_STATE, stratify=labels
    )

    clf = FinancialSentimentClassifier()
    clf.fit(X_train, y_train)
    report = clf.evaluate(X_test, y_test)
    log.info("\n%s", report)
    clf.save()
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate daily sentiment scores
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    clf: FinancialSentimentClassifier,
    date_col: str = "date",
    headline_col: str = "headline",
) -> pd.DataFrame:
    """
    Given a DataFrame of news articles, compute per-day sentiment features:
        - sentiment_positive, sentiment_negative, sentiment_neutral (mean proba)
        - sentiment_score (positive − negative, range -1 to +1)
        - headline_count
    """
    probas = clf.predict_proba(news_df[headline_col].tolist())
    classes = clf.pipeline.classes_

    proba_df = pd.DataFrame(probas, columns=classes, index=news_df.index)
    proba_df[date_col] = news_df[date_col].values

    daily = proba_df.groupby(date_col).agg(
        sentiment_positive=("positive", "mean") if "positive" in classes else (classes[0], "mean"),
        sentiment_negative=("negative", "mean") if "negative" in classes else (classes[-1], "mean"),
        headline_count=(classes[0], "count"),
    ).reset_index()

    pos_col = "positive" if "positive" in classes else classes[0]
    neg_col = "negative" if "negative" in classes else classes[-1]
    daily["sentiment_score"] = (
        proba_df.groupby(date_col)[pos_col].mean()
      - proba_df.groupby(date_col)[neg_col].mean()
    ).values

    daily.index = pd.to_datetime(daily[date_col])
    return daily.drop(columns=[date_col])


# ─────────────────────────────────────────────────────────────────────────────
# Demo with synthetic data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_headlines = [
        "Apple reports record quarterly earnings beating analyst estimates",
        "Fed raises interest rates amid inflation concerns",
        "Tech stocks decline sharply on recession fears",
        "Goldman Sachs upgrades Tesla to buy with strong price target",
        "Oil prices crash as demand outlook weakens globally",
    ]
    sample_labels = ["positive", "neutral", "negative", "positive", "negative"]

    clf = FinancialSentimentClassifier()
    clf.fit(sample_headlines, sample_labels)

    for h in sample_headlines[:3]:
        result = clf.score_headline(h)
        print(f"\nHeadline : {h}")
        print(f"Sentiment: {result['label']}")
        print(f"Probas   : {result['probabilities']}")
