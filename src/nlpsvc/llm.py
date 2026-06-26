"""LLM zero-shot classifier — the baseline the fine-tuned model competes against.

MockLLM is deterministic and key-free (classifies by cue words, with a realistic error rate
and simulated latency/cost). ClaudeLLM makes a real zero-shot call. Same `.classify` contract.
"""
from __future__ import annotations

import os
import random
import re

from .config import LABELS, LEXICON, MODEL, PRICE


def _cue_label(text: str):
    w = set(re.findall(r"[a-z]+", text.lower()))
    scores = {lbl: len(w & set(LEXICON[lbl])) for lbl in LABELS}
    best = max(scores.values())
    top = [l for l, s in scores.items() if s == best and best > 0]
    return top[0] if len(top) == 1 else None


class MockLLM:
    name = "llm_zero_shot(mock)"
    ERROR_RATE = 0.12          # LLMs misclassify narrow-domain tickets some of the time
    LATENCY_MS = 540           # network + generation latency per call

    def classify(self, text: str) -> dict:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        pred = _cue_label(text) or rng.choice(LABELS)
        if rng.random() < self.ERROR_RATE:
            pred = rng.choice([l for l in LABELS if l != pred])
        return {"label": pred, "latency_ms": self.LATENCY_MS,
                "input_tokens": 110 + len(text) // 4, "output_tokens": 3}


class ClaudeLLM:
    name = MODEL

    def __init__(self, model: str = MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def classify(self, text: str) -> dict:
        import time

        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model, max_tokens=8,
            system=f"Classify the support ticket into exactly one label: {', '.join(LABELS)}. "
                   "Reply with only the label.",
            messages=[{"role": "user", "content": text}])
        raw = next((b.text for b in resp.content if b.type == "text"), "").strip().lower()
        pred = next((l for l in LABELS if l in raw), LABELS[0])
        return {"label": pred, "latency_ms": round((time.perf_counter() - t0) * 1000),
                "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * PRICE["input"] + output_tokens / 1e6 * PRICE["output"]


def get_llm():
    return ClaudeLLM() if os.getenv("ANTHROPIC_API_KEY") else MockLLM()
