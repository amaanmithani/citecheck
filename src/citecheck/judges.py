"""Judges estimate the probability that evidence supports a claim."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import httpx


class Judge(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """For each (evidence, claim) return P(evidence supports claim) in [0, 1]."""
        ...


def _key(evidence: str, claim: str) -> str:
    return hashlib.sha256(f"{evidence}\x00{claim}".encode()).hexdigest()


class CachedJudge:
    """Memoises another judge; metrics re-ask the same pairs a lot."""

    def __init__(self, inner: Judge) -> None:
        self.inner = inner
        self.name = inner.name
        self._cache: dict[str, float] = {}

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        keys = [_key(e, c) for e, c in pairs]
        missing = [(k, p) for k, p in zip(keys, pairs, strict=True) if k not in self._cache]
        if missing:
            uniq = dict(missing)
            for k, s in zip(uniq, self.inner.score(list(uniq.values())), strict=True):
                self._cache[k] = s
        return [self._cache[k] for k in keys]


class NLIJudge:
    """A cross-encoder NLI model. Evidence longer than the model's window is
    split into overlapping chunks and the claim's best-supported chunk wins,
    so a supporting sentence deep inside a long document isn't truncated away."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
        max_evidence_tokens: int = 384,
        stride: int = 192,
        batch_size: int = 16,
        max_chunks: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        # CPU measured faster than MPS for these small cross-encoders.
        self.device = device or "cpu"
        self.model.to(self.device)
        labels = {i: str(lbl).lower() for i, lbl in self.model.config.id2label.items()}
        ent = [i for i, lbl in labels.items() if "entail" in lbl]
        if len(ent) != 1:
            raise ValueError(f"{model}: can't find the entailment label in {labels}")
        self.ent = ent[0]
        self.max_ev, self.stride, self.bs, self.max_chunks = max_evidence_tokens, stride, batch_size, max_chunks

    def _chunks(self, evidence: str) -> list[str]:
        ids = self.tok(evidence, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_ev:
            return [evidence]
        out = []
        for s in range(0, len(ids), self.stride):
            out.append(str(self.tok.decode(ids[s : s + self.max_ev])))
            if s + self.max_ev >= len(ids) or len(out) >= self.max_chunks:
                break
        return out

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        flat: list[tuple[int, str, str]] = []
        for i, (ev, claim) in enumerate(pairs):
            for ch in self._chunks(ev):
                flat.append((i, ch, claim))
        # Similar lengths together: less padding per batch.
        flat.sort(key=lambda x: len(x[1]) + len(x[2]))
        best = [0.0] * len(pairs)
        with self.torch.no_grad():
            for b in range(0, len(flat), self.bs):
                batch = flat[b : b + self.bs]
                enc = self.tok(
                    [x[1] for x in batch],
                    [x[2] for x in batch],
                    truncation="only_first",
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.ent].tolist()
                for (i, _, _), p in zip(batch, probs, strict=True):
                    best[i] = max(best[i], float(p))
        return best


JUDGE_PROMPT = """You check citations. Decide whether the EVIDENCE fully supports the CLAIM:
every fact in the claim must be stated in, or directly implied by, the evidence.
Background knowledge doesn't count. Answer with exactly one word: yes or no.

EVIDENCE:
{evidence}

CLAIM:
{claim}"""


class LLMJudge:
    """An LLM over any OpenAI-compatible endpoint (Ollama, ModelMux, OpenAI).
    Returns 1.0 for yes, 0.0 for no, or 0.5 when the reply is neither."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "unused",
        timeout: float = 120.0,
        max_evidence_chars: int = 12_000,
    ) -> None:
        self.name = f"llm:{model}"
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.max_chars = max_evidence_chars

    def _one(self, evidence: str, claim: str) -> float:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3,
            "messages": [
                {"role": "user", "content": JUDGE_PROMPT.format(evidence=evidence[: self.max_chars], claim=claim)}
            ],
        }
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"] or ""
        word = re.sub(r"[^a-z]", "", text.strip().lower().split()[0] if text.strip() else "")
        return 1.0 if word.startswith("yes") else 0.0 if word.startswith("no") else 0.5

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [self._one(e, c) for e, c in pairs]
