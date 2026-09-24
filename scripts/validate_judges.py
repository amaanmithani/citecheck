"""Validate judges against AttributionBench human labels.

    uv run python scripts/validate_judges.py --judges nli-xsmall,nli-small,llm --llm-sample 400

Writes results/judges.json. NLI judges run on every example; the LLM judge,
being slower, on a fixed label-stratified sample of each split (stated in the
output)."""

import argparse
import json
import time
from pathlib import Path

from citecheck.attrbench import evaluate, load, sample
from citecheck.judges import CachedJudge, LLMJudge, NLIJudge

MODELS = {"nli-xsmall": "cross-encoder/nli-deberta-v3-xsmall", "nli-small": "cross-encoder/nli-deberta-v3-small"}

ap = argparse.ArgumentParser()
ap.add_argument("--judges", default="nli-xsmall,nli-small,llm")
ap.add_argument("--llm-model", default="llama3.1:8b")
ap.add_argument("--llm-sample", type=int, default=400)
ap.add_argument("--nli-sample", type=int, default=600)
ap.add_argument("--out", default="results/judges.json")
a = ap.parse_args()

cache = Path("data")
splits = {s: load(s, cache) for s in ("dev", "test", "test_ood")}
out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {"judges": {}}
out["protocol"] = (
    "threshold chosen on the AttributionBench dev split (maximising balanced accuracy) and frozen; "
    "metrics on the in-distribution test and out-of-distribution test splits; claim judged against the "
    "concatenated references; 95% CIs by bootstrap (1000 resamples)"
)
for name in a.judges.split(","):
    if name == "llm":
        judge, n = CachedJudge(LLMJudge(a.llm_model)), a.llm_sample
    else:
        judge, n = CachedJudge(NLIJudge(MODELS[name])), a.nli_sample
    dev = sample(splits["dev"], n)
    tests = {k: sample(splits[k], n) for k in ("test", "test_ood")}
    t0 = time.time()
    res = evaluate(judge, dev, tests)
    res["seconds"] = round(time.time() - t0, 1)
    res["sampled"] = n is not None
    out["judges"][name] = res
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    t, o = res["test"], res["test_ood"]
    print(
        f"{name}: thr={res['threshold_from_dev']} test BA={t['balanced_accuracy']} k={t['kappa']} | "
        f"ood BA={o['balanced_accuracy']} k={o['kappa']} ({res['seconds']}s)",
        flush=True,
    )
