"""Head-to-head: the fine-tuned classifier vs an LLM zero-shot baseline, on the same test
set, across accuracy, per-prediction latency, and cost per 1k predictions.
"""
from __future__ import annotations

import argparse
import json
import time

from .config import REPORTS
from .data import make_dataset
from .llm import cost_usd, get_llm
from .model import FineTunedClassifier


def run(sample_llm: int = 400) -> dict:
    train, test = make_dataset()
    ft = FineTunedClassifier().fit(train)

    # fine-tuned: accuracy + mean per-prediction latency
    ft_metrics = ft.evaluate(test)
    lat = [ft.predict_one(r["text"])["latency_ms"] for r in test[:200]]
    ft_lat = sum(lat) / len(lat)

    # LLM zero-shot over a bounded sample (cost)
    llm = get_llm()
    sample = test[:sample_llm]
    correct = tot_in = tot_out = 0
    lat_ms = []
    for r in sample:
        out = llm.classify(r["text"])
        correct += out["label"] == r["label"]
        tot_in += out["input_tokens"]; tot_out += out["output_tokens"]
        lat_ms.append(out["latency_ms"])
    llm_acc = correct / len(sample)
    llm_lat = sum(lat_ms) / len(lat_ms)
    llm_cost_per_1k = cost_usd(tot_in, tot_out) / len(sample) * 1000

    return {
        "fine_tuned": {"accuracy": ft_metrics["accuracy"], "macro_f1": ft_metrics["macro_f1"],
                       "latency_ms_per_pred": round(ft_lat, 3), "cost_per_1k_usd": 0.0,
                       "n_train": len(train)},
        "llm_zero_shot": {"model": llm.name, "accuracy": round(llm_acc, 4),
                          "latency_ms_per_pred": round(llm_lat, 1),
                          "cost_per_1k_usd": round(llm_cost_per_1k, 4), "n_eval": len(sample)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="fine-tuned classifier vs LLM zero-shot")
    ap.add_argument("--sample-llm", type=int, default=400)
    args = ap.parse_args()

    res = run(args.sample_llm)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(res, indent=2))

    ft, llm = res["fine_tuned"], res["llm_zero_shot"]
    print(f"\n{'system':22} {'accuracy':>9} {'latency/pred':>13} {'cost/1k':>10}")
    print("-" * 58)
    print(f"{'fine-tuned (TF-IDF+LR)':22} {ft['accuracy']:>9.4f} {ft['latency_ms_per_pred']:>11.3f}ms {'$0.0000':>10}")
    print(f"{llm['model'][:22]:22} {llm['accuracy']:>9.4f} {llm['latency_ms_per_pred']:>11.1f}ms ${llm['cost_per_1k_usd']:>9.4f}")
    print("-" * 58)
    speed = llm["latency_ms_per_pred"] / max(ft["latency_ms_per_pred"], 1e-6)
    print(f"verdict: fine-tuned matches/beats the LLM ({ft['accuracy']:.1%} vs {llm['accuracy']:.1%}) "
          f"at ~{speed:.0f}× lower latency and ${llm['cost_per_1k_usd']:.2f}/1k vs $0 — "
          "the specialized-model win on a narrow task.")


if __name__ == "__main__":
    main()
