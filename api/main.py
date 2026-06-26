"""FastAPI serving the fine-tuned classifier. Trains on startup if no artifact exists.

  POST /classify {text}  -> {label, confidence, latency_ms}
  GET  /labels
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nlpsvc.config import LABELS, MODEL_PATH
from nlpsvc.data import make_dataset
from nlpsvc.model import FineTunedClassifier

app = FastAPI(title="NLP Classifier Service", version="0.1.0")

if MODEL_PATH.exists():
    _model = FineTunedClassifier.load()
else:
    train, _ = make_dataset()
    _model = FineTunedClassifier().fit(train)
    _model.save()


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "labels": LABELS}


@app.get("/labels")
def labels():
    return {"labels": LABELS}


@app.post("/classify")
def classify(req: ClassifyRequest):
    return _model.predict_one(req.text)
