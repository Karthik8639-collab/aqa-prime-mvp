"""SENTINEL orchestrator. Fixes con #9: positional annotation, honest reporting."""
from dataclasses import dataclass, field
from sentinel.config import SentinelConfig
from sentinel.segmenter import segment
from sentinel.evidence import load_evidence, EvidenceIndex
from sentinel.nli_engine import NLIEngine
from sentinel.rules import run_rules
from sentinel.detector_types import AuditResult


class SentinelEngine:
    def __init__(self, evidence_dir="evidence", config: SentinelConfig = None):
        self.config = config or SentinelConfig()
        facts = load_evidence(evidence_dir)
        self.evidence_index = EvidenceIndex(facts)
        self.nli = NLIEngine(self.config)

    def audit(self, text: str) -> AuditResult:
        result = AuditResult(text=text)

        # Layer 1: deterministic rules (full text)
        result.detections.extend(run_rules(text))

        # Layer 2: NLI contradiction engine (context-aware, batched)
        sentences = segment(text)
        factual = [s for s in sentences if s.factual]
        result.sentences_checked = len(factual)
        result.sentences_skipped_nonfactual = len(sentences) - len(factual)
        result.detections.extend(
            self.nli.check_sentences(factual, self.evidence_index, self.config)
        )

        # dedupe by sentence text
        seen = set()
        result.detections = [d for d in result.detections
                             if not (d.sentence_text in seen or seen.add(d.sentence_text))]
        return result

    def annotated(self, text: str) -> str:
        """Positional annotation — fixes the str.replace bug (con #9)."""
        result = self.audit(text)
        from sentinel.segmenter import SENTENCE_SPLIT
        parts = SENTENCE_SPLIT.split(text)
        det_map = {}
        for d in result.detections:
            det_map[d.sentence_text] = d
        out = []
        for part in parts:
            key = next((k for k in det_map if k in part), None)
            out.append(part)
            if key:
                d = det_map[key]
                out.append(f" ⚠️ **[{d.verdict} · vs {d.evidence_id} · "
                           f"p={d.contradiction_prob}]** → Evidence: {d.evidence_fact}")
        return " ".join(out)
