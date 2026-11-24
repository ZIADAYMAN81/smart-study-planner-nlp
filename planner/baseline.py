"""Classical baseline: TF-IDF features with logistic regression, on the same balanced subsets."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .metrics import evaluate


def train_baseline(train: dict, test: dict) -> dict:
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=1000, C=4.0))
    model.fit(train["text"], train["label"])
    metrics = evaluate(test["label"], model.predict(test["text"]))
    metrics["model"] = "TF-IDF + logistic regression"
    return metrics
