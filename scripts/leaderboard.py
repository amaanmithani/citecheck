"""Citation leaderboard on HAGRID: which retrieval setup gives answers whose
citations actually hold up?

For a fixed sample of HAGRID dev questions, each configuration retrieves 5
passages from the pool of all HAGRID dev passages, llama3.1:8b writes a cited
answer, and citecheck scores it with the judge validated in judges.json.

    uv run python scripts/leaderboard.py --n 100

Writes results/leaderboard.json (and results/answers.jsonl with every answer).
"""

import argparse
import json
import random
import time
from pathlib import Path

from citecheck.judges import CachedJudge, NLIJudge
from citecheck.metrics import check
from citecheck.rag import BM25, Dense, Generator, Passage, Reranker, rrf

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--gen-model", default="llama3.1:8b")
ap.add_argument("--embed-model", default="mxbai-embed-large")
ap.add_argument("--judge", default="nli-small")
ap.add_argument("--data", default="data/hagrid_dev.jsonl")
a = ap.parse_args()

MODELS = {"nli-xsmall": "cross-encoder/nli-deberta-v3-xsmall", "nli-small": "cross-encoder/nli-deberta-v3-small"}
judges = json.loads(Path("results/judges.json").read_text())["judges"]
threshold = judges[a.judge]["threshold_from_dev"]
judge = CachedJudge(NLIJudge(MODELS[a.judge]))

rows = [json.loads(line) for line in Path(a.data).read_text().splitlines()]
pool: dict[str, Passage] = {}
for r in rows:
    for q in r["quotes"]:
        pool[q["docid"]] = Passage(q["docid"], q["text"])
passages = list(pool.values())
qs = random.Random(20260924).sample(rows, a.n)

print(f"{len(passages)} passages, {len(qs)} questions; judge {a.judge} @ {threshold}", flush=True)
bm25 = BM25(passages)
dense = Dense(passages, a.embed_model)
rr = Reranker()
gen = Generator(a.gen_model)

CONFIGS = {
    "bm25": lambda q: bm25.search(q, a.k),
    "dense": lambda q: dense.search(q, a.k),
    "hybrid": lambda q: rrf(bm25.search(q, 20), dense.search(q, 20), top=a.k),
    "bm25+rerank": lambda q: rr.rerank(q, bm25.search(q, 20), a.k),
    "dense+rerank": lambda q: rr.rerank(q, dense.search(q, 20), a.k),
    "hybrid+rerank": lambda q: rr.rerank(q, rrf(bm25.search(q, 20), dense.search(q, 20), top=20), a.k),
}

results = {}
answers = []
for name, retrieve in [*CONFIGS.items(), ("oracle (gold passages)", None)]:
    t0 = time.time()
    rec, prec, unsup, abstain, retr_recall, n_cit = [], [], [], 0, [], []
    for r in qs:
        gold = {q["docid"] for q in r["quotes"]}
        got = (
            [Passage(q["docid"], q["text"]) for q in r["quotes"]][: a.k]
            if retrieve is None
            else [p for p, _ in retrieve(r["query"])]
        )
        retr_recall.append(len(gold & {p.id for p in got}) / len(gold))
        ans = gen.answer(r["query"], got)
        answers.append({"config": name, "query_id": r["query_id"], "answer": ans, "passages": [p.id for p in got]})
        if "passages don't say" in ans.lower():
            abstain += 1
            continue
        rep = check(ans, [p.text for p in got], judge, threshold)
        rec.append(rep.citation_recall)
        prec.append(rep.citation_precision)
        unsup.append(rep.unsupported_rate)
        n_cit.append(sum(len(s.citations) for s in rep.sentences))
    m = len(rec) or 1
    results[name] = {
        "questions": len(qs),
        "answered": len(rec),
        "abstained": abstain,
        "retrieval_recall_at_k": round(sum(retr_recall) / len(retr_recall), 4),
        "citation_recall": round(sum(rec) / m, 4),
        "citation_precision": round(sum(prec) / m, 4),
        "unsupported_sentence_rate": round(sum(unsup) / m, 4),
        "citations_per_answer": round(sum(n_cit) / m, 2),
        "seconds": round(time.time() - t0, 1),
    }
    print(name, results[name], flush=True)
    Path("results/leaderboard.json").write_text(
        json.dumps(
            {
                "what": "citation quality of RAG answers by retrieval configuration",
                "method": f"{a.n} HAGRID dev questions (seeded sample); retrieval over all {len(passages)} "
                f"HAGRID dev passages, "
                f"top {a.k}; answers by {a.gen_model} (temperature 0); citations scored by {a.judge} at the "
                f"threshold calibrated on AttributionBench dev ({threshold}); metrics averaged over answered "
                "questions; "
                "'oracle' uses HAGRID's gold passages as an upper bound",
                "judge_error_note": "judge accuracy vs humans is in results/judges.json; "
                "treat differences smaller than the "
                "judge's error as ties",
                "configs": results,
            },
            indent=2,
        )
        + "\n"
    )
Path("results/answers.jsonl").write_text("\n".join(json.dumps(x) for x in answers) + "\n")
