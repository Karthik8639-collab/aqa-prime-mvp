from fastapi import FastAPI
from pydantic import BaseModel
from sentinel.engine import SentinelEngine

app = FastAPI(title="SENTINEL", version="2.0")
engine = SentinelEngine()

class AuditRequest(BaseModel):
    text: str

@app.post("/audit")
def audit(req: AuditRequest):
    r = engine.audit(req.text)
    return {
        "summary": r.summary(),
        "sentences_checked": r.sentences_checked,
        "skipped_nonfactual": r.sentences_skipped_nonfactual,
        "detections": [{
            "sentence": d.sentence_text,
            "verdict": d.verdict,
            "evidence": d.evidence_fact,
            "evidence_id": d.evidence_id,
            "probability": d.contradiction_prob,
            "severity": d.severity,
            "layer": d.layer,
        } for d in r.detections],
        "disclaimer": ("Only claims contradicting the loaded evidence base "
                       "or deterministic rules are flagged. Absence of flags "
                       "is not verification."),
    }

# run: uvicorn api.main:app --reload
