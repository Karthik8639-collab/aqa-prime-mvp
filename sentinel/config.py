"""All thresholds in one place. Values below are STARTING points —
run eval/run_eval.py toibrate against the benchmark before shipping."""
from dataclasses import dataclass


@dataclass
class SentinelConfig:
    # NLI cross-encoder (contradiction probability thresholds)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    CONTRADICTION_FLAG: float = 0.80   # >= this: auto-flag
    CONTRADICTION_REVIEW: float = 0.60 # between: human review queue

    # Sentence relevance: skip NLI on sentences with no checkable content
    MIN_SENTENCE_WORDS: int = 4

    # Number of candidate evidence facts per sentence (top-k retrieval)
    NLI_TOP_K: int = 5

    # Self-consistency layer (novel-hallucination catching)
    consistency_samples: int = 5
    consistency_temp: float = 0.9
    consistency_disagree: float = 0.50  # fraction of samples disagreeing = flag

    # Rule layer is always deterministic: threshold = 1.0 by definition
