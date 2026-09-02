"""
Layer 3: Self-consistency check for novel hallucinations.

Rationale: when an LLM invents a fact, resampling at temperature produces
inconsistent answers (different invented names, dates, numbers). Real facts
are stable. This catches errors that exist NOWHERE in the evidence base —
the single biggest gap in v1 and in most vault-based guardrails.

This runs ONLY on high-risk sentences (those the NLI layer marked NEUTRAL
but which contain checkable specifics: dates, numbers, names, citations).
"""
import re
from sentinel.detector_types import Detection

SPECIFICITY = re.compile(
    r'\b(19|20)\d{2}\b'                       # years
    r'|\b\d+(?:\.\d+)?\s?(?:%|mg|kg|km|MHz|GHz|nm)\b'  # quantities
    r'|\b[A-Z][a-z]+ (?:et al\.|University|Institute|Journal)\b'  # citations
)


def extract_specifics(sentence: str) -> list:
    """Pull out the verifiable specifics (dates, numbers, names) for comparison."""
    return SPECIFICITY.findall(sentence) + re.findall(r'\b(?:19|20)\d{2}\b', sentence)


def consistency_flags(original: str, resamples: list) -> dict:
    """
    Compare key specifics across k resamples.
    Returns agreement stats. A specific (a date, a figure) that appears in
    < half of resamples is UNSTABLE -> probable hallucination.
    """
    if not resamples:
        return {"stable": True, "unstable_specifics": []}
    original_specifics = set(extract_specifics(original))
    unstable = []
    for spec in original_specifics:
        support = sum(1 for r in resamples if spec in r)
        if support / len(resamples) < 0.5:
            unstable.append({"specific": spec, "support": f"{support}/{len(resamples)}"})
    return {"stable": not unstable, "unstable_specifics": unstable}
