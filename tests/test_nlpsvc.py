"""Data, model, LLM baseline, and the head-to-head benchmark (key-free)."""
from nlpsvc.benchmark import run
from nlpsvc.config import LABELS
from nlpsvc.data import make_dataset
from nlpsvc.llm import MockLLM, cost_usd
from nlpsvc.model import FineTunedClassifier


def test_dataset_shapes_and_labels():
    train, test = make_dataset(n_train=300, n_test=100, seed=1)
    assert len(train) == 300 and len(test) == 100
    assert set(r["label"] for r in train) <= set(LABELS)


def test_fine_tuned_is_accurate():
    train, test = make_dataset(n_train=600, n_test=200, seed=2)
    ft = FineTunedClassifier().fit(train)
    assert ft.evaluate(test)["accuracy"] > 0.9          # on-distribution -> strong
    out = ft.predict_one("I want a refund on my invoice")
    assert out["label"] in LABELS and 0 <= out["confidence"] <= 1


def test_model_roundtrip(tmp_path, monkeypatch):
    import nlpsvc.config as cfg
    monkeypatch.setattr(cfg, "MODEL_PATH", tmp_path / "m.pkl")
    monkeypatch.setattr(cfg, "ARTIFACTS", tmp_path)
    import nlpsvc.model as m
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "m.pkl")
    monkeypatch.setattr(m, "ARTIFACTS", tmp_path)
    train, _ = make_dataset(n_train=200, n_test=10, seed=3)
    FineTunedClassifier().fit(train).save()
    assert FineTunedClassifier.load().predict(["my package tracking is wrong"])[0] in LABELS


def test_mock_llm_and_cost():
    out = MockLLM().classify("my login throws an error")
    assert out["label"] in LABELS and out["latency_ms"] > 0
    assert cost_usd(1000, 100) > 0


def test_benchmark_fine_tuned_wins_on_cost_and_latency():
    res = run(sample_llm=200)
    ft, llm = res["fine_tuned"], res["llm_zero_shot"]
    assert ft["cost_per_1k_usd"] == 0.0 and llm["cost_per_1k_usd"] > 0       # LLM costs money
    assert ft["latency_ms_per_pred"] < llm["latency_ms_per_pred"]            # and is slower
    assert ft["accuracy"] >= llm["accuracy"]                                 # FT matches/beats on-domain
