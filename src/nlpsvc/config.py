"""Labels, lexicons, pricing, paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("NLP_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pkl"
REPORTS = ROOT / "reports"

MODEL = os.getenv("NLP_LLM_MODEL", "claude-opus-4-8")
PRICE = {"input": 5.0, "output": 25.0}  # USD / 1M tokens

LABELS = ["billing", "technical", "account", "shipping"]
LEXICON = {
    "billing": ["invoice", "charge", "refund", "payment", "subscription", "price", "receipt", "card"],
    "technical": ["error", "bug", "crash", "broken", "loading", "timeout", "login", "install"],
    "account": ["profile", "email", "username", "delete", "settings", "permissions", "deactivate"],
    "shipping": ["delivery", "tracking", "package", "shipment", "arrive", "courier", "address", "delayed"],
}
