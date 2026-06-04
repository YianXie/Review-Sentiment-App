from pathlib import Path
from typing import Literal

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
clf = joblib.load(MODELS_DIR / "clf.joblib")


def predict(review: str) -> Literal["positive", "negative", "neutral"]:
    review_tfidf = vectorizer.transform([review])
    prediction = clf.predict(review_tfidf)[0]
    return prediction
