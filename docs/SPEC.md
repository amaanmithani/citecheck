# citecheck — spec

Checks whether the citations in a generated answer actually support the
sentences they're attached to, and measures how far that check itself can be
trusted against human judgements.

## Goals (v1)

| # | Capability | Done when |
|---|---|---|
| G1 | Split an answer into sentences and map each to its citations (`[1]`, `[2][3]`, `[1, 2]`) | unit tests on messy real-world formatting |
| G2 | Judges that score "does this evidence support this claim": NLI cross-encoders (long evidence chunked, max over chunks) and an LLM judge over any OpenAI-compatible endpoint (Ollama, ModelMux) | same interface; results cached |
| G3 | ALCE-style metrics per answer: **citation recall** (sentence supported by its citations together), **citation precision** (each citation needed or sufficient), **unsupported-sentence rate** | unit tests with a fake judge |
| G4 | **Judge validation against humans** on AttributionBench: threshold calibrated on the dev split only; balanced accuracy, macro-F1 and Cohen's κ reported on the in-distribution and out-of-distribution test splits, per source dataset, with bootstrap CIs | committed results JSON |
| G5 | Leaderboard: answers generated under several retrieval configurations, scored by the validated judge | committed results JSON |
| G6 | API `POST /check` and a page showing sentence-level support | demo |

## Non-goals
Checking the factual truth of claims (only whether the cited text supports them); multilingual input.

## Honesty rules
- Thresholds are chosen on dev and never tuned on test.
- Every number in the README comes from a committed script and results file.
- The judge's own error rate is reported next to every metric it produces.
