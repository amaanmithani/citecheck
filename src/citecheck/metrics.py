"""Citation metrics for one answer, following ALCE (Gao et al., 2023).

- A sentence is **supported** if the concatenation of its cited documents
  entails it (citation recall).
- A citation is **relevant** if it supports the sentence alone, or if the
  sentence is supported jointly and removing that citation breaks the support.
  Otherwise it's irrelevant (citation precision = relevant / all citations).
- Sentences without citations count as unsupported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .judges import Judge
from .segment import Sentence, split


@dataclass
class SentenceResult:
    text: str
    citations: list[int]
    supported: bool
    support_prob: float
    relevant: list[int] = field(default_factory=list)
    irrelevant: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)  # cited numbers with no such document


@dataclass
class Report:
    sentences: list[SentenceResult]
    citation_recall: float  # share of sentences supported by their citations
    citation_precision: float  # share of citations that are relevant
    unsupported_rate: float
    uncited_sentences: int
    judge: str
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _joined(docs: list[str], ids: list[int]) -> str:
    return "\n\n".join(docs[i - 1] for i in ids)


def check(answer: str, docs: list[str], judge: Judge, threshold: float = 0.5) -> Report:
    """Score an answer whose citations [n] refer to docs[n-1]."""
    sents: list[Sentence] = split(answer)
    results: list[SentenceResult] = []
    n_cites = n_relevant = n_supported = uncited = 0
    for s in sents:
        valid = [c for c in s.citations if 1 <= c <= len(docs)]
        missing = [c for c in s.citations if c not in valid]
        if not valid:
            uncited += not s.citations
            results.append(SentenceResult(s.text, s.citations, False, 0.0, [], [], missing))
            n_cites += len(missing)
            continue
        joint = judge.score([(_joined(docs, valid), s.text)])[0]
        supported = joint >= threshold
        relevant: list[int] = []
        irrelevant: list[int] = []
        alone = judge.score([(docs[c - 1], s.text) for c in valid])
        without = (
            judge.score([(_joined(docs, [x for x in valid if x != c]), s.text) for c in valid])
            if supported and len(valid) > 1
            else [0.0] * len(valid)
        )
        for c, a, w in zip(valid, alone, without, strict=True):
            if a >= threshold or (supported and len(valid) > 1 and w < threshold) or (supported and len(valid) == 1):
                relevant.append(c)
            else:
                irrelevant.append(c)
        n_supported += supported
        n_cites += len(valid) + len(missing)
        n_relevant += len(relevant)
        results.append(SentenceResult(s.text, s.citations, supported, round(joint, 4), relevant, irrelevant, missing))
    total = len(results) or 1
    return Report(
        sentences=results,
        citation_recall=n_supported / total,
        citation_precision=(n_relevant / n_cites) if n_cites else 0.0,
        unsupported_rate=1 - n_supported / total,
        uncited_sentences=uncited,
        judge=judge.name,
        threshold=threshold,
    )
