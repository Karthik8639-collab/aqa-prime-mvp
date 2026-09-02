from dataclasses import dataclass, field


@dataclass
class Detection:
    sentence: str
    source_id: str          # evidence id or rule id
    evidence: str           # the TRUE fact, or the rule's correction
    prob: float
    severity: str           # critical / high / review
    layer: str              # "rule" or "nli"
    verdict: str


@dataclass
class AuditResult:
    text: str
    detections: list = field(default_factory=list)
    sentences_checked: int = 0
    sentences_skipped: int = 0

    @property
    def flagged(self):
        return [d for d in self.detections if d.severity != "review"]

    @property
    def for_review(self):
        return [d for d in self.detections if d.severity == "review"]

    def summary(self) -> str:
        return (f"Checked {self.sentences_checked} factual sentences "
                f"({self.sentences_skipped} skipped as non-factual). "
                f"Flagged: {len(self.flagged)} · Review: {len(self.for_review)}")
