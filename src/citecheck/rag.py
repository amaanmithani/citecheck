"""A small RAG pipeline for the citation leaderboard: retrievers, a reranker
and a citing generator over an OpenAI-compatible endpoint."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "and",
        "or",
        "for",
        "on",
        "at",
        "by",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "as",
        "from",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "did",
        "do",
        "does",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Passage:
    id: str
    text: str


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.p = list(passages)
        self.docs = [tokenize(x.text) for x in self.p]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = len(self.docs[i])
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.p[i], scores[i]) for i in order]


class Dense:
    """Cosine search over embeddings from an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self, passages: Sequence[Passage], model: str, base_url: str = "http://localhost:11434/v1", batch: int = 64
    ) -> None:
        self.p = list(passages)
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=300)
        self.vecs: list[list[float]] = []
        for i in range(0, len(self.p), batch):
            self.vecs.extend(self._embed([x.text for x in self.p[i : i + batch]]))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.http.post("/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        out = [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return [_unit(v) for v in out]

    def search(self, query: str, k: int) -> list[tuple[Passage, float]]:
        q = self._embed([query])[0]
        scores = [sum(a * b for a, b in zip(q, v, strict=True)) for v in self.vecs]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.p[i], scores[i]) for i in order]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def rrf(*rankings: list[tuple[Passage, float]], k: int = 60, top: int = 10) -> list[tuple[Passage, float]]:
    """Reciprocal rank fusion."""
    score: dict[str, float] = {}
    by_id: dict[str, Passage] = {}
    for ranking in rankings:
        for rank, (p, _) in enumerate(ranking):
            score[p.id] = score.get(p.id, 0.0) + 1 / (k + rank + 1)
            by_id[p.id] = p
    order = sorted(score, key=lambda i: -score[i])[:top]
    return [(by_id[i], score[i]) for i in order]


class Reranker:
    """A cross-encoder relevance model (MS MARCO)."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()

    def rerank(self, query: str, cands: list[tuple[Passage, float]], k: int) -> list[tuple[Passage, float]]:
        with self.torch.no_grad():
            enc = self.tok(
                [query] * len(cands),
                [p.text for p, _ in cands],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            s = self.model(**enc).logits.squeeze(-1).tolist()
        order = sorted(range(len(cands)), key=lambda i: -s[i])[:k]
        return [(cands[i][0], s[i]) for i in order]


GEN_PROMPT = """Answer the question using ONLY the numbered passages below.
After every sentence, cite the passage(s) that support it, like [1] or [2][3].
Don't cite a passage that doesn't support the sentence. If the passages don't
contain the answer, say "The passages don't say." and nothing else.
Keep the answer to at most 4 sentences.

{passages}

Question: {question}
Answer:"""


class Generator:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1") -> None:
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=300)

    def answer(self, question: str, passages: Sequence[Passage]) -> str:
        block = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
        r = self.http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "seed": 7,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": GEN_PROMPT.format(passages=block, question=question)}],
            },
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()
