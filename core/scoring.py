import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util


def keyword_overlap_bonus(text_a: str, text_b: str) -> float:
    """Small cosine boost based on keyword Jaccard overlap (bounded)."""
    def tokens(s: str) -> set[str]:
        words = re.findall(r"[\w/\-]+", s.lower())
        return {w for w in words if len(w) >= 3}

    ta = tokens(text_a)
    tb = tokens(text_b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    jacc = inter / union
    return min(0.02, jacc * 0.04)


def detect_lexical_contradictions(req_units: List[str], feat_units: List[str]) -> bool:
    """Toy lexical contradiction detector for demo sanity checks (png vs exe, immediacy vs delayed)."""
    req_text = "\n".join(req_units).lower()
    feat_text = "\n".join(feat_units).lower()
    if any(tok in req_text for tok in ["png", "jpeg", "jpg"]) and (".exe" in feat_text or " exe" in feat_text):
        return True
    pos_immediate = any(p in req_text for p in ["сразу отображается", "немедленно", "мгновенно"])
    neg_display = any(p in feat_text for p in ["не отображается", "до перезагрузки"])
    if pos_immediate and neg_display:
        return True
    return False


def compute_soft_f1_score(req_units: List[str], feat_units: List[str], model: SentenceTransformer, sim_threshold: float) -> Tuple[float, float, float, float]:
    """Compute soft F1, hit_rate (recall), precision with cosine threshold matching."""
    if not req_units or not feat_units:
        return 0.0, 0.0, 0.0, 0.0
    req_emb = model.encode(req_units, convert_to_tensor=True, normalize_embeddings=True)
    feat_emb = model.encode(feat_units, convert_to_tensor=True, normalize_embeddings=True)
    sim_rf = util.cos_sim(req_emb, feat_emb)  # [R, F]

    # recall: fraction of requirements covered by any feature >= threshold
    covered_req = 0
    for i in range(sim_rf.size(0)):
        if float(sim_rf[i].max().item()) >= sim_threshold:
            covered_req += 1
    recall = covered_req / max(1, len(req_units))

    # precision: fraction of feature units that match any requirement >= threshold
    sim_fr = sim_rf.transpose(0, 1)  # [F, R]
    matched_feat = 0
    for i in range(sim_fr.size(0)):
        if float(sim_fr[i].max().item()) >= sim_threshold:
            matched_feat += 1
    precision = matched_feat / max(1, len(feat_units))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # hit_rate == recall in this formulation
    hit_rate = recall
    return f1, hit_rate, precision, recall


