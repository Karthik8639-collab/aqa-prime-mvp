"""v2 segmentation: sentence-level units WITH context, fixing con #5.
Each unit carries its neighbors so NLI sees the full stance context."""
import re
from dataclasses import dataclass

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
# Discourse markers that signal non-factual sentences (skip candidate)
OPINION_MARKERS = re.compile(
    r'\b(i think|i believe|in my opinion|maybe|perhaps|probably|'
    r'it seems|allegedly|reportedly|some say)\b', re.I
)
QUESTION = re.compile(r'\?\s*$')


@dataclass
class Sentence:
    text: str
    index: int
    context_before: str   # previous sentence (negation scope, stance)
    context_after: str    # next sentence
    factual: bool         # whether it's a checkable factual assertion


def segment(text: str) -> list:
    raw = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    sentences = []
    for i, s in enumerate(raw):
        factual = not bool(OPINION_MARKERS.search(s)) and not QUESTION.search(s)
        sentences.append(Sentence(
            text=s,
            index=i,
            context_before=raw[i - 1] if i > 0 else "",
            context_after=raw[i + 1] if i < len(raw) - 1 else "",
            factual=factual,
        ))
    return sentences
