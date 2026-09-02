// evidence/science.json — NOTE: facts are TRUE claims we defend
{
  "domain": "science",
  "facts": [
    {
      "id": "sci-001",
      "fact": "Antibiotics treat bacterial infections, not viral infections.",
      "topic_tags": ["medicine", "biology", "pharmacy"],
      "source": "WHO / standard pharmacology"
    },
    {
      "id": "sci-002",
      "fact": "The Earth orbits the Sun, completing one revolution per year.",
      "topic_tags": ["astronomy", "physics"],
      "source": "elementary astronomy"
    },
    {
      "id": "sci-003",
      "fact": "Sound requires a material medium and cannot propagate through a vacuum.",
      "topic_tags": ["physics", "acoustics"],
      "source": "classical mechanics"
    },
    {
      "id": "sci-004",
      "fact": "Carbon forms four covalent bonds in stable compounds (tetravalence).",
      "topic_tags": ["chemistry"],
      "source": "organic chemistry"
    },
    {
      "id": "sci-005",
      "fact": "In a vacuum, all objects fall at the same rate regardless of mass.",
      "topic_tags": ["physics"],
      "source": "Galilean mechanics"
    },
    {
      "id": "sci-006",
      "fact": "Insulin lowers blood glucose; glucagon raises it.",
      "topic_tags": ["medicine", "endocrinology"],
      "source": "physiology"
    },
    {
      "id": "sci-007",
      "fact": "Photosynthesis consumes carbon dioxide and releases oxygen.",
      "topic_tags": ["biology", "botany"],
      "source": "plant physiology"
    },
    {
      "id": "sci-008",
      "fact": "Energy cannot be created or destroyed, only transformed.",
      "topic_tags": ["physics", "thermodynamics"],
      "source": "first law of thermodynamics"
    },
    {
      "id": "sci-009",
      "fact": "No heat engine can exceed 100 percent efficiency.",
      "topic_tags": ["thermodynamics", "engineering"],
      "source": "second law of thermodynamics"
    },
    {
      "id": "sci-010",
      "fact": "Mitochondrial DNA is inherited almost exclusively through the maternal line.",
      "topic_tags": ["genetics", "biology"],
      "source": "genetics"
    }
  ]
}
# sentinel/evidence.py
"""Evidence base: curated TRUE facts. A hallucination is any claim
that CONTRADICTS one of these — regardless of how the error is phrased
or whether we've seen that specific error before. This kills con #1."""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_evidence(dir_path: str = "evidence", domains=None) -> list:
    facts = []
    for path in sorted(Path(dir_path).glob("*.json")):
        if domains and path.stem not in domains:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for f in data.get("facts", []):
            for field in ("id", "fact", "topic_tags"):
                if field not in f:
                    raise ValueError(f"{path.name}: fact missing '{field}'")
            facts.append(f)
    ids = [f["id"] for f in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence IDs")
    return facts


class EvidenceIndex:
    """
    Two-stage retrieval (kills con #3 properly):
      Stage 1: embedding retrieval — top-k most relevant facts per sentence.
               This is a RELEVANCE gate, not a keyword gate. Paraphrases
               retrieve their facts because embeddings capture topic.
      Stage 2: NLI cross-encoder decides entail/contradict/neutral.
    """
    def __init__(self, facts: list, retriever_model: str = "all-MiniLM-L6-v2"):
        self.facts = facts
        self.retriever = SentenceTransformer(retriever_model)
        self._fact_vectors = self.retriever.encode(
            [f["fact"] for f in facts], normalize_embeddings=True
        )

    def retrieve(self, sentence: str, top_k: int = 5) -> list:
        vec = self.retriever.encode([sentence], normalize_embeddings=True)[0]
        sims = self._fact_vectors @ vec
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.facts[i] for i in top_idx]
