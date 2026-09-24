from collections.abc import Sequence

from citecheck.judges import CachedJudge
from citecheck.metrics import check


class KeywordJudge:
    """Supports a claim when every word of it longer than 3 letters appears in the evidence."""

    name = "keyword"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls += len(pairs)
        out = []
        for ev, claim in pairs:
            words = [w.strip(".,").lower() for w in claim.split() if len(w.strip(".,")) > 3]
            out.append(1.0 if all(w in ev.lower() for w in words) else 0.0)
        return out


DOCS = ["Paris is the capital city of France.", "France borders Spain and Italy.", "Bananas are yellow."]


def test_supported_and_precision():
    r = check("Paris capital France [1]. France borders Spain [2][3].", DOCS, KeywordJudge())
    a, b = r.sentences
    assert a.supported and a.relevant == [1]
    assert b.supported and b.relevant == [2] and b.irrelevant == [3]
    assert r.citation_recall == 1.0 and r.citation_precision == 2 / 3


def test_joint_support_needs_both():
    r = check("Paris capital France borders Spain [1][2].", DOCS, KeywordJudge())
    s = r.sentences[0]
    # Neither doc supports it alone, together they do: both relevant.
    assert s.supported and s.relevant == [1, 2]


def test_unsupported_uncited_and_missing():
    r = check("Paris is purple [1]. Uncited claim here. Bad ref [9].", DOCS, KeywordJudge())
    assert [s.supported for s in r.sentences] == [False, False, False]
    assert r.uncited_sentences == 1 and r.sentences[2].missing == [9]
    assert r.unsupported_rate == 1.0 and r.citation_precision == 0.0
    assert "sentences" in r.to_dict()


def test_empty_answer():
    r = check("", DOCS, KeywordJudge())
    assert r.citation_recall == 0.0 and r.citation_precision == 0.0


def test_cached_judge_dedups():
    inner = KeywordJudge()
    j = CachedJudge(inner)
    j.score([("a", "b"), ("a", "b"), ("c", "d")])
    j.score([("a", "b")])
    assert inner.calls == 2
