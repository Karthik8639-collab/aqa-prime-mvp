# 🛡️ SENTINEL — Claim-Level Verification Engine

**v2.0 · "VERITAS-core"**

SENTINEL audits AI-generated (or any) text at the **claim level** and flags
assertions that **contradict verified knowledge** — regardless of how the
error is phrased, and regardless of whether that specific error was ever
seen before.

## Why v2 exists (the honest history)

v1 ("AQA-Prime") matched text against a curated list of *known lies* using
word similarity. Independent expert review identified fatal flaws:

| Flaw in v1 | Fix in v2 |
|---|---|
| Vault stored **lies** → only caught previously-seen errors | Evidence base stores **truths** → catches *any* claim that contradicts them, including errors we never anticipated |
| MiniLM cosine similarity **cannot distinguish a sentence from its negation** | NLI cross-encoder (`nli-deberta-v3-base`) outputs entailment/neutral/contradiction natively |
| Keyword gates did all the work; paraphrases slipped through | Retrieval is **semantic relevance**, and NLI reads both texts jointly — paraphrase-robust by construction |
| Correct *debunkings* were flagged ("It is a myth that…") | Directional entailment check: only text that **asserts** the negation of a fact is flagged |
| No evaluation — numbers were vibes | `eval/run_eval.py` with a labeled benchmark: precision / recall / F1 per category |
| Clause-splitting destroyed negation scope | Sentence-level checking **with preceding context** passed to the NLI model |
| Not real-time | Batched NLI engine + streaming interceptor (`sentinel/streaming.py` in roadmap) |

## How it works
