"""
NLI contradiction engine — fixes cons #2, #3, #4, #6.

Model: cross-encoder/nli-deberta-v3-base (≈184M params, runs on CPU).
It takes (premise, hypothesis) PAIRS and outputs entailment/neutral/
contradiction — it reads both texts jointly, so negation and stance
are handled by the model itself, not by string heuristics.

Key design: we ask "does the TEXT ENTAIL the NEGATION of a known fact?"
  - Text: "Antibiotics are the best treatment for the flu"
  - Fact: "Antibiotics treat bacterial infections, not viral infections"
  - Negated fact hypothesis: "Antibiotics are NOT effective against viruses"
  -> text entails negated-fact => CONTRADICTION with evidence => FLAG.

Debunking is naturally safe:
  - Text: "It is a myth that antibiotics cure colds"
  -> text CONTRADICTS the negated-fact => text agrees with evidence => PASS.
"""
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentinel.config import SentinelConfig
from sentinel.detector_types import Detection


class NLIEngine:
    def __init__(self, config: SentinelConfig = None):
        self.config = config or SentinelConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nli_model
        )
        self.model.eval()
        # label order for nli-deberta-v3-base: 0=entailment,1=neutral,2=contradiction
        self.ENTAIL, self.NEUTRAL, self.CONTRADICT = 0, 1, 2

    @torch.no_grad()
    def _score_pairs(self, pairs: list) -> list:
        """Batched scoring — fixes con #8 (no per-item encode)."""
        if not pairs:
            return []
        inputs = self.tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            truncation=True, max_length=256, padding=True, return_tensors="pt"
        )
        logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1)
        return probs  # shape (n_pairs, 3)

    def check_sentences(self, sentences: list, evidence_index, config=None) -> list:
        """
        Full pipeline for a batch of Sentence objects:
          retrieve top-k facts -> build (text, negated-fact) pairs ->
          batch NLI -> flag contradictions.
        Negation of the fact is done by template insertion; the NLI model
        handles the linguistic composition.
        """
        config = config or self.config
        pairs, meta = [], []

        for sent in sentences:
            if not sent.factual or len(sent.text.split()) < config.MIN_SENTENCE_WORDS:
                continue
            # Context-rich premise: fixes con #5 — NLI sees stance scope
            premise = (sent.context_before + " " + sent.text).strip()
            for fact in evidence_index.retrieve(sent.text, config.NLI_TOP_K):
                hypothesis = f"It is false that {fact['fact'][0].lower() + fact['fact'][1:]}"
                pairs.append((premise, hypothesis))
                meta.append({"sentence": sent, "fact": fact})

        if not pairs:
            return []

        probs = self._score_pairs(pairs)
        detections = []
        for (premise, _), m, p in zip(pairs, meta, probs):
            # p[:, ENTAIL] = probability text ASSERTS the negated fact
            #               = probability text CONTRADICTS the true fact
            contra_prob = float(p[config.ENTAIL if False else 0])  # placeholder
            contra_prob = float(p[0])
            if contra_prob >= config.CONTRADICTION_FLAG:
                detections.append(Detection(
                    sentence_text=m["sentence"].text,
                    evidence_id=m["fact"]["id"],
                    evidence_fact=m["fact"]["fact"],
                    source=m["fact"].get("source", "curated"),
                    contradiction_prob=round(contra_prob, 3),
                    severity="critical" if contra_prob >= 0.9 else "high",
                    layer="nli",
                    verdict="CONTRADICTS_EVIDENCE",
                ))
            elif contra_prob >= config.CONTRADICTION_REVIEW:
                detections.append(Detection(
                    sentence_text=m["sentence"].text,
                    evidence_id=m["fact"]["id"],
                    evidence_fact=m["fact"]["fact"],
                    source=m["fact"].get("source", "curated"),
                    contradiction_prob=round(contra_prob, 3),
                    severity="review",
                    layer="nli",
                    verdict="NEEDS_HUMAN_REVIEW",
                ))
        return detections

# sentinel/detector_types.py
from dataclasses import dataclass

@dataclass
class Detection:
    sentence_text: str
    evidence_id: str
    evidence_fact: str
    source: str
    contradiction_prob: float
    severity: str
    layer: str
    verdict: str

@dataclass
class AuditResult:
    text: str
    sentences_checked: int = 0
    sentences_skipped_nonfactual: int = 0
    detections: list = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []

    @property
    def clean(self):
        return not any(d.severity != "review" for d in self.detections)

    def summary(self):
        flagged = [d for d in self.detections if d.severity != "review"]
        reviews = [d for d in self.detections d.severity == "review"]
        lines = [
            f"Checked {self.sentences_checked} factual sentences "
            f"({self.sentences_skipped_nonfactual} skipped as non-factual)."
        ]
        lines.append(f"Flagged: {len(flagged)} | Needs human review: {len(reviews)}")
        return "\n".join(lines)
