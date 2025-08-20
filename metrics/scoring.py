from typing import Tuple
from sentence_transformers import SentenceTransformer
from .model import embedding_score
from core.scoring import keyword_overlap_bonus


def compute_document_alignment_score(md_text: str, feat_text: str, model: SentenceTransformer) -> Tuple[float, float, float]:
    """Return (agg_base, cos, score).

    Definitions:
      - agg_base: base document-level similarity (max of symmetric variants). In current implementation
        doc_cos = fwd = rev, so agg_base == doc_cos. Kept as max() for future asymmetric variants.
      - cos: bounded cosine after a small keyword-overlap bonus (see core.scoring.keyword_overlap_bonus).
      - score: cos scaled to percentage [0..100].

    Practical meaning of agg_base:
      Used by evaluation to relax coverage requirements when global doc alignment is high (≥0.82, ≥0.85, ≥0.90).
    """
    doc_cos = embedding_score(md_text, feat_text)
    # Forward: md sentences as a whole against feature text; reverse symmetric — kept simple here
    fwd = doc_cos
    rev = doc_cos
    agg_base = max(doc_cos, fwd, rev)
    bonus = keyword_overlap_bonus(md_text, feat_text)
    cos = min(1.0, agg_base + bonus)
    score = cos * 100.0
    return agg_base, cos, score


