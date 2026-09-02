Benchmark harness. This is what converts SENTINEL from a demo into a system.

eval/benchmark.json format — 200 sentences, each labeled:
  {"text": "...", "label": "HALLUCINATED" | "CLEAN", "category": "adversarial_paraphrase" | "negation" | "straightforward_error" | "clean_tricky" | "clean_plain"}

Category design (this is the honest test):
  - straightforward_error (40): the false claim, plainly stated
  - adversarial_paraphrase (40): false claim, unfamiliar phrasing/synonyms
  - negation (20): text DENIES the myth — must NOT be flagged (the hard test)
  - clean_plain (40): true statements
  - clean_tricky (40): true statements that LOOK like famous errors
    ("Antibiotics revolutionized bacterial pneumonia treatment" — clean!)
  - opinion/context (20): non-factual sentences — must be skipped, not flagged

Run: python eval/run_eval.py  →  prints precision/recall/F1 per category.
"""
import json
from pathlib import Path
from sentinel.engine import SentinelEngine


def run(benchmark_path="eval/benchmark.json"):
    engine = SentinelEngine()
    data = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))

    tp = fp = fn = tn = 0
    per_category = {}

    for item in data["sentences"]:
        result = engine.audit(item["text"])
        flagged = any(d.severity != "review" for d in result.detections)
        truth = item["label"] == "HALLUCINATED"

        cat = per_category.setdefault(item["category"], {"tp":0,"fp":0,"fn":0,"tn":0})
        if flagged and truth: tp += 1; cat["tp"] += 1
        elif flagged and not truth: fp += 1; cat["fp"] += 1
        elif not flagged and truth: fn += 1; cat["fn"] += 1
        else: tn += 1; cat["tn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"OVERALL  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    for cat, c in sorted(per_category.items()):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0
        f = 2*p*r/(p+r) if p+r else 0
        print(f"  {cat:28s} P={p:.2f} R={r:.2f} F1={f:.2f}")

    print("\nCalibration guidance:")
    print(" - If FP high on clean_tricky → raise CONTRADICTION_FLAG")
    print(" - If FN high on adversarial_paraphrase → raise NLI_TOP_K or lower flag threshold")
    print(" - Negation F1 below 0.8 means stance handling needs work — fix before shipping")


if __name__ == "__main__":
    run()
