# 📝 NLP Service: fine-tuned classifier vs LLM zero-shot

> A deployed **text-classification service** built on a **fine-tuned** small model (TF-IDF +
> LogisticRegression), benchmarked head-to-head against an **LLM zero-shot** baseline on
> **accuracy, latency, and cost**. The senior call most teams get wrong: for a narrow,
> high-volume task, a small specialized model **beats the LLM on every axis**. Self-contained
> (synthetic labeled data); `ANTHROPIC_API_KEY` swaps the mock baseline for real Claude.

Reaching for an LLM to classify support tickets is the expensive default. This shows the
trade-off with numbers, when you have labels and volume, fine-tune the small model and serve
it for ~$0 at sub-millisecond latency.

---

## The head-to-head (`nlpsvc-benchmark`)

```
$ nlpsvc-benchmark
system                  accuracy  latency/pred    cost/1k
----------------------------------------------------------
fine-tuned (TF-IDF+LR)    1.0000       0.222ms    $0.0000
llm_zero_shot(mock)       0.8775     540.0ms     $0.6810
----------------------------------------------------------
verdict: fine-tuned matches/beats the LLM (100.0% vs 87.8%) at ~2432× lower latency
         and $0.68/1k vs $0 — the specialized-model win on a narrow task.
```

> **Honest basis:** these numbers are on **synthetic, templated data** (each ticket is built
> from its own label's cue-words, so the 100% reflects a highly-separable demo set, not a
> production claim), and the LLM figures (87.8% / 540ms / $0.68) are **modeled by `MockLLM`,
> not measured**, set `ANTHROPIC_API_KEY` to benchmark against real Claude.

On a **narrow, labeled task** the fine-tuned model wins **accuracy** (trained on-distribution),
**latency** (no network, no generation), and **cost** ($0 vs per-call tokens). The LLM's value
is zero-shot flexibility on *new* tasks, but where you have labels and volume, specialize.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions, never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

nlpsvc-benchmark                 # fine-tuned vs LLM zero-shot (mock baseline, offline)

# serve the fine-tuned classifier (trains on first start)
$PY -m uvicorn api.main:app --port 8000
curl -s localhost:8000/classify -H 'content-type: application/json' \
  -d '{"text":"I was overcharged on my last invoice"}'   # -> {"label":"billing", ...}

export ANTHROPIC_API_KEY=sk-ant-...
nlpsvc-benchmark                 # baseline now hits real Claude zero-shot
```

---

## Architecture

```
data.py     synthetic labeled support tickets (4-way intent) → train / test
   │
model.py    fine-tuned TF-IDF + LogisticRegression classifier (save/load, predict + confidence)
   │                                   llm.py   LLM zero-shot (mock / Claude)
   ▼                                      │
benchmark.py  fine-tuned vs LLM → accuracy · latency/pred · cost/1k → verdict
   │
api/main.py  POST /classify (serves the fine-tuned model)
```

---

## Repo layout

```
nlp-service/
├── src/nlpsvc/
│   ├── data.py       synthetic labeled tickets + train/test split
│   ├── model.py      fine-tuned TF-IDF+LogReg classifier (fit/predict/save/load)
│   ├── llm.py        LLM zero-shot baseline (mock + Claude) + token pricing
│   ├── benchmark.py  fine-tuned vs LLM: accuracy · latency · cost  (CLI)
│   └── config.py     labels, lexicons, pricing, paths
├── api/main.py       FastAPI /classify · /labels
├── tests/            data · model accuracy · roundtrip · benchmark trade-off — 5 cases
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built and deployed a fine-tuned text-classification service (TF-IDF + LogReg) and
> benchmarked it against an LLM zero-shot baseline on accuracy, latency, and cost, showing
> the specialized model matches/beats the LLM at ~N× lower latency and $0 vs per-call cost on
> a narrow high-volume task. The "do I even need an LLM here" judgment.*

## License
MIT (`LICENSE`).
