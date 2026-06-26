"""The fine-tuned classifier: TF-IDF + LogisticRegression. Small, fast, CPU-only,
deployable — the specialized model we're putting up against an LLM zero-shot baseline."""
from __future__ import annotations

import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS, MODEL_PATH


class FineTunedClassifier:
    name = "fine_tuned"

    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("lr", LogisticRegression(max_iter=1000, C=8.0)),
        ])

    def fit(self, train: list[dict]):
        self.pipe.fit([r["text"] for r in train], [r["label"] for r in train])
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipe.predict(texts))

    def predict_one(self, text: str) -> dict:
        t0 = time.perf_counter()
        proba = self.pipe.predict_proba([text])[0]
        classes = self.pipe.classes_
        i = int(proba.argmax())
        return {"label": classes[i], "confidence": round(float(proba[i]), 4),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    def evaluate(self, test: list[dict]) -> dict:
        pred = self.predict([r["text"] for r in test])
        gold = [r["label"] for r in test]
        return {"accuracy": round(accuracy_score(gold, pred), 4),
                "macro_f1": round(f1_score(gold, pred, average="macro"), 4)}

    def save(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipe, f)

    @classmethod
    def load(cls):
        obj = cls()
        with open(MODEL_PATH, "rb") as f:
            obj.pipe = pickle.load(f)
        return obj
