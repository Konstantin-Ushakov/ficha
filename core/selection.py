from typing import List
from sentence_transformers import SentenceTransformer, util


def select_topk_feature_units(req_units: List[str], feat_units: List[str], model: SentenceTransformer) -> List[str]:
    if not req_units or not feat_units:
        return feat_units
    req_emb = model.encode(req_units, convert_to_tensor=True, normalize_embeddings=True)
    feat_emb = model.encode(feat_units, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(feat_emb, req_emb)
    best_per_feat = sim.max(dim=1).values.cpu().tolist()
    ranked_indices = sorted(range(len(feat_units)), key=lambda i: best_per_feat[i], reverse=True)
    import math
    k = max(1, min(len(feat_units), math.ceil(len(req_units) * 1.1)))
    selected_idx = set(ranked_indices[:k])
    return [feat_units[i] for i in range(len(feat_units)) if i in selected_idx]


