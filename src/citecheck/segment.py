"""Split a cited answer into sentences and the citations attached to each."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# [1] · [1, 2] · [1-3] · [1–3] · [1][2] (adjacent groups are matched one at a time)
CITE = re.compile(r"\[(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]")
_ABBREV = {
    "e.g",
    "i.e",
    "etc",
    "vs",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "st",
    "u.s",
    "u.k",
    "inc",
    "ltd",
    "no",
    "fig",
    "al",
}


@dataclass
class Sentence:
    text: str  # sentence text with citation markers removed
    citations: list[int] = field(default_factory=list)  # 1-based, in order of first appearance
    start: int = 0
    end: int = 0


def parse_group(group: str) -> list[int]:
    """'1, 3-5' -> [1, 3, 4, 5]. Ranges are capped at 50 to bound bad input."""
    out: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a < 50:
                out.extend(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def _is_abbrev(text: str, dot: int) -> bool:
    """Is the period at text[dot] the end of an abbreviation or a decimal?"""
    if dot + 1 < len(text) and text[dot + 1].isdigit() and dot > 0 and text[dot - 1].isdigit():
        return True
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:dot])
    if not m:
        return False
    word = m.group(1).lower()
    return word in _ABBREV or (len(word) == 1 and word.isalpha())


def split(answer: str) -> list[Sentence]:
    """Split into sentences. Citation markers that directly follow a sentence's
    final punctuation ("... capital.[1] Next") belong to that sentence."""
    sentences: list[Sentence] = []
    start, i, n = 0, 0, len(answer)
    while i < n:
        ch = answer[i]
        if ch in ".!?" and not (ch == "." and _is_abbrev(answer, i)):
            j = i + 1
            while j < n and answer[j] in ".!?\"'”’)":
                j += 1
            # absorb trailing citation groups
            while True:
                k = j
                while k < n and answer[k] == " ":
                    k += 1
                m = CITE.match(answer, k)
                if not m:
                    break
                j = m.end()
            if j >= n or answer[j].isspace():
                _emit(answer, start, j, sentences)
                start = j
                i = j
                continue
        i += 1
    _emit(answer, start, n, sentences)
    return sentences


def _emit(answer: str, a: int, b: int, out: list[Sentence]) -> None:
    raw = answer[a:b]
    if not raw.strip():
        return
    cites: list[int] = []
    for m in CITE.finditer(raw):
        for c in parse_group(m.group(1)):
            if c not in cites:
                cites.append(c)
    text = re.sub(r"\s+", " ", CITE.sub("", raw)).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    if text:
        lead = len(raw) - len(raw.lstrip())
        out.append(Sentence(text=text, citations=cites, start=a + lead, end=b))
