from typing import Dict, List, Tuple
import os
from sentence_transformers import SentenceTransformer
from .nli_then import extract_then_statements
from core.models import nli_entailment_prob
import numpy as np


def compute_nli_hit_rate(
    req_units: List[str],
    feature_text: str,
    model: SentenceTransformer,
    entailment_threshold: float = 0.70,
) -> Tuple[float, Dict[str, List]]:
    """NLI-based coverage: fraction of requirements entailed by any Then/And statement.

    Definition:
      For each requirement unit, we consider all Then/And statements from the feature file(s)
      (no GW context in this metric). If max entailment probability ≥ entailment_threshold,
      the requirement is counted as entailed.

    Returns:
      (nli_hit_rate, details) where details contains:
        - entailed_idx: indices of entailed requirements
        - not_entailed_idx: indices not entailed
        - best_entailment: list of floats with max entailment probability per requirement
        - best_pairs: list of tuples (req_index, best_then_text, best_entail_prob)
        - best_top3_pairs: list of tuples (req_index, then_text, entail_prob) — up to 3 per requirement
        - worst_pairs: list of tuples (req_index, worst_then_text, worst_entail_prob)
    Notes:
      This metric complements kw_hit_rate; it is stricter and sensitive to phrasing.
    """
    details: Dict[str, List] = {
        "entailed_idx": [],
        "not_entailed_idx": [],
        "best_entailment": [],
        "best_pairs": [],
        "best_top3_pairs": [],
        "worst_pairs": [],
    }
    if not req_units:
        return 0.0, details
    thens = extract_then_statements(feature_text)
    if not thens:
        return 0.0, details
    n = len(req_units)
    best = [0.0] * n

    # Precompute embeddings for Then/And once to cheaply preselect candidates by cosine
    try:
        thens_emb = model.encode(thens, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        thens_emb = None

    top_k = int(os.environ.get("NLI_TOPK", "10"))
    bottom_k = int(os.environ.get("NLI_BOTTOMK", "5"))
    if top_k < 1:
        top_k = 1
    if bottom_k < 0:
        bottom_k = 0

    for i, r in enumerate(req_units):
        max_e = 0.0
        best_then = ""
        min_e = 1.0
        worst_then = ""
        evaluated: List[tuple[str, float]] = []

        # Select a subset of candidate Then/And lines: top-K most similar (by embeddings),
        # plus bottom-K least similar for a meaningful "worst" signal. This caps NLI calls.
        candidate_indices: List[int]
        if thens_emb is not None:
            try:
                r_emb = model.encode([r], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
                sims = np.dot(thens_emb, r_emb)
                order = np.argsort(sims)[::-1]
                top_idx = order[: min(top_k, len(thens))]
                bot_idx = order[::-1][: min(bottom_k, len(thens))] if bottom_k > 0 else np.array([], dtype=int)
                candidate_indices = list(dict.fromkeys(list(top_idx) + list(bot_idx)))  # de-dup, keep order
            except Exception:
                candidate_indices = list(range(len(thens)))
        else:
            candidate_indices = list(range(len(thens)))

        # Evaluate NLI only on selected candidates, with early stop on success
        for j in candidate_indices:
            t = thens[j]
            try:
                e = nli_entailment_prob(r, t)
            except Exception:
                continue
            evaluated.append((t, e))
            if e > max_e:
                max_e = e
                best_then = t
            if e < min_e:
                min_e = e
                worst_then = t
            if max_e >= entailment_threshold:
                # Early stop once we have a satisfying entailment
                break

        # Prepare report details (top-3 by entailment among evaluated set)
        if evaluated:
            evaluated.sort(key=lambda x: x[1], reverse=True)
            for (t, e) in evaluated[:3]:
                details["best_top3_pairs"].append((i, t, e))

        best[i] = max_e
        details["best_pairs"].append((i, best_then, max_e))
        details["worst_pairs"].append((i, worst_then, min_e))
        if max_e >= entailment_threshold:
            details["entailed_idx"].append(i)
        else:
            details["not_entailed_idx"].append(i)

    details["best_entailment"] = best
    nli_hit_rate = len(details["entailed_idx"]) / max(1, n)
    return nli_hit_rate, details


def get_description() -> str:
    return (
        "nli_hit_rate — доля требований .md, логически подразумеваемых хоть одним Then/And (по NLI-порогy). "
        "Чувствительна к формулировкам и направлению ожиданий."
    )


def get_detailed_fix() -> str:
    return (
        "Добавьте явные Then/And, которые прямо фиксируют ожидаемый результат (включая направление/величину), "
        "или перефразируйте требования/шаги, чтобы entailment по NLI стал ≥ порога."
    )

