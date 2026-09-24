"""Validate judges against human attribution labels (AttributionBench).

Protocol: the decision threshold is chosen on the dev split only (maximising
balanced accuracy) and then frozen; every reported number comes from the
in-distribution test split or the out-of-distribution test split.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .judges import Judge

BASE = "https://huggingface.co/datasets/osunlp/AttributionBench/resolve/main/"
SPLITS = {
    "dev": "dev_all_subset_balanced.jsonl",
    "test": "test_all_subset_balanced.jsonl",
    "test_ood": "test_ood_all_subset_balanced.jsonl",
}


@dataclass
class Example:
    id: str
    question: str
    claim: str
    evidence: str
    label: int  # 1 = attributable
    source: str


def load(split: str, cache: Path) -> list[Example]:
    path = cache / SPLITS[split]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + SPLITS[split], path)
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.append(
            Example(
                id=r["id"],
                question=r.get("question") or "",
                claim=r["claim"],
                evidence="\n\n".join(r.get("references") or []),
                label=1 if r["attribution_label"] == "attributable" else 0,
                source=r["src_dataset"],
            )
        )
    return out


def sample(exs: list[Example], n: int | None, seed: int = 0) -> list[Example]:
    """Deterministic, label-stratified subsample (for slow judges)."""
    if n is None or n >= len(exs):
        return exs
    rng = random.Random(seed)
    pos = [e for e in exs if e.label]
    neg = [e for e in exs if not e.label]
    k = n // 2
    return rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(n - k, len(neg)))


def claim_text(e: Example) -> str:
    # The claim alone: citecheck judges sentences, not question-answer pairs.
    return e.claim


def confusion(y: list[int], p: list[int]) -> tuple[int, int, int, int]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p, strict=True))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p, strict=True))
    return tp, tn, fp, fn


def metrics(y: list[int], p: list[int]) -> dict[str, float]:
    tp, tn, fp, fn = confusion(y, p)
    n = len(y) or 1
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {
        "balanced_accuracy": (tpr + tnr) / 2,
        "macro_f1": (f1_pos + f1_neg) / 2,
        "kappa": kappa,
        "accuracy": po,
        "recall_attributable": tpr,
        "recall_not_attributable": tnr,
    }


def best_threshold(y: list[int], probs: list[float]) -> float:
    """The threshold maximising balanced accuracy on (dev) data."""
    cands = sorted(set([0.5] + [round(x, 4) for x in probs]))
    best, best_ba = 0.5, -1.0
    for t in cands:
        ba = metrics(y, [int(q >= t) for q in probs])["balanced_accuracy"]
        if ba > best_ba:
            best, best_ba = t, ba
    return best


def bootstrap(y: list[int], p: list[int], reps: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    rng = random.Random(seed)
    n = len(y)
    acc: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        m = metrics([y[i] for i in idx], [p[i] for i in idx])
        for k, v in m.items():
            acc[k].append(v)
    return {
        k: [round(sorted(v)[int(0.025 * reps)], 4), round(sorted(v)[int(0.975 * reps) - 1], 4)] for k, v in acc.items()
    }


def evaluate(judge: Judge, dev: list[Example], tests: dict[str, list[Example]]) -> dict[str, object]:
    dev_probs = judge.score([(e.evidence, claim_text(e)) for e in dev])
    t = best_threshold([e.label for e in dev], dev_probs)
    out: dict[str, object] = {
        "judge": judge.name,
        "threshold_from_dev": t,
        "dev_n": len(dev),
        "dev_balanced_accuracy": round(
            metrics([e.label for e in dev], [int(q >= t) for q in dev_probs])["balanced_accuracy"], 4
        ),
    }
    for name, exs in tests.items():
        probs = judge.score([(e.evidence, claim_text(e)) for e in exs])
        y, p = [e.label for e in exs], [int(q >= t) for q in probs]
        by_src: dict[str, dict[str, float]] = {}
        for src in sorted({e.source for e in exs}):
            idx = [i for i, e in enumerate(exs) if e.source == src]
            m = metrics([y[i] for i in idx], [p[i] for i in idx])
            by_src[src] = {"n": len(idx), "balanced_accuracy": round(m["balanced_accuracy"], 4)}
        out[name] = {
            "n": len(exs),
            **{k: round(v, 4) for k, v in metrics(y, p).items()},
            "ci95": bootstrap(y, p),
            "by_source": by_src,
        }
    return out
