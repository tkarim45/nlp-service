"""Synthetic labeled support tickets (4-way intent). Deterministic train/test split — the
labeled data a team would fine-tune a small classifier on instead of paying an LLM per call."""
from __future__ import annotations

import random

from .config import LABELS, LEXICON

_TEMPLATES = [
    "I need help with my {a}.",
    "There's a problem with the {a} and the {b}.",
    "Can you check the {a}? The {b} isn't right.",
    "My {a} keeps failing — {b} issues since yesterday.",
    "Question about {a} and {b}, thanks.",
    "Please look into the {a}; the {b} is wrong.",
]


def _text(label: str, rng: random.Random) -> str:
    a, b = rng.sample(LEXICON[label], 2)
    return rng.choice(_TEMPLATES).format(a=a, b=b)


def make_dataset(n_train: int = 1200, n_test: int = 400, seed: int = 7):
    rng = random.Random(seed)
    rows = [{"text": _text(lbl := rng.choice(LABELS), rng), "label": lbl}
            for _ in range(n_train + n_test)]
    return rows[:n_train], rows[n_train:]
