import json
from collections.abc import Sequence
from pathlib import Path

from citecheck import attrbench as ab


def test_metrics_and_kappa():
    m = ab.metrics([1, 1, 0, 0], [1, 1, 0, 0])
    assert m["balanced_accuracy"] == 1 and m["kappa"] == 1 and m["macro_f1"] == 1
    chance = ab.metrics([1, 0, 1, 0], [1, 1, 1, 1])
    assert chance["balanced_accuracy"] == 0.5 and abs(chance["kappa"]) < 1e-9
    assert ab.metrics([], [])["accuracy"] == 0


def test_best_threshold_picks_separating_point():
    y = [0, 0, 1, 1]
    probs = [0.1, 0.3, 0.6, 0.9]
    t = ab.best_threshold(y, probs)
    assert 0.3 < t <= 0.6


def test_bootstrap_bounds():
    ci = ab.bootstrap([1, 0] * 50, [1, 0] * 50, reps=200)
    assert ci["balanced_accuracy"] == [1.0, 1.0]


def test_load_sample_and_evaluate(tmp_path: Path):
    rows = []
    for i in range(20):
        lab = i % 2 == 0
        rows.append(
            {
                "id": str(i),
                "question": "q",
                "claim": "sky blue" if lab else "sky green",
                "references": ["The sky is blue."],
                "attribution_label": "attributable" if lab else "not attributable",
                "src_dataset": "A" if i < 10 else "B",
            }
        )
    for f in ab.SPLITS.values():
        (tmp_path / f).write_text("\n".join(json.dumps(r) for r in rows))
    dev = ab.load("dev", tmp_path)
    assert len(dev) == 20 and dev[0].label == 1 and dev[1].label == 0
    s = ab.sample(dev, 6)
    assert len(s) == 6 and sum(e.label for e in s) == 3
    assert ab.sample(dev, None) == dev

    class J:
        name = "j"

        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [0.9 if "blue" in c else 0.1 for _, c in pairs]

    res = ab.evaluate(J(), dev, {"test": dev, "test_ood": dev})
    assert res["test"]["balanced_accuracy"] == 1.0
    assert set(res["test"]["by_source"]) == {"A", "B"}
