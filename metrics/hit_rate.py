from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util


def compute_hit_rate(
    req_units: List[str],
    feat_units: List[str],
    model: SentenceTransformer,
    sim_threshold: float,
) -> Tuple[float, Dict[str, List]]:
    """Compute classic hit_rate between requirement units and feature units.

    Definition:
      A requirement unit is considered covered if there exists at least one feature unit
      whose cosine similarity (on normalized sentence embeddings) is ≥ sim_threshold.

    Returns:
      (hit_rate, details) where details contains:
        - hits_idx: indices of covered requirements
        - misses_idx: indices of uncovered requirements
        - matches: list of tuples (req_index, feat_index, similarity)
    """
    details: Dict[str, List] = {"hits_idx": [], "misses_idx": [], "matches": []}
    if not req_units or not feat_units:
        return 0.0, details
    req_emb = model.encode(req_units, convert_to_tensor=True, normalize_embeddings=True)
    feat_emb = model.encode(feat_units, convert_to_tensor=True, normalize_embeddings=True)
    sim_mat = util.cos_sim(req_emb, feat_emb)
    hits = 0
    for i in range(sim_mat.size(0)):
        row = sim_mat[i]
        max_val, max_idx = row.max(dim=0)
        sim_val = float(max_val.item())
        j = int(max_idx.item())
        details["matches"].append((i, j, sim_val))
        if sim_val >= sim_threshold:
            hits += 1
            details["hits_idx"].append(i)
        else:
            details["misses_idx"].append(i)
    hit_rate = hits / max(1, len(req_units))
    return hit_rate, details


def get_description() -> str:
    return (
        "hit_rate — доля требований из .md, для которых нашлась хотя бы одна строка .feature "
        "с косинусной близостью ≥ SIM_THRESHOLD. Показывает базовое семантическое покрытие."
    )


def get_detailed_fix() -> str:
    return (
        "Поднимайте hit_rate добавлением явных Then/And для каждого смыслового пункта .md; "
        "используйте те же сущности и формулировки. При необходимости дробите длинные пункты в .md."
    )

