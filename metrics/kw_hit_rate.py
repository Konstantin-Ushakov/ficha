import re
from typing import Dict, List, Tuple


def compute_kw_hit_rate(
    req_units: List[str],
    feat_units: List[str],
    best_match_indices: List[int],
    min_shared_tokens: int = 2,
) -> Tuple[float, Dict[str, List]]:
    """Keyword-overlap hit rate for interpretability.

    Definition:
      For each requirement unit i we take its best-matching feature unit index j (by cosine),
      and count a 'keyword hit' if the sets of alphanumeric tokens share at least
      `min_shared_tokens` elements. This does not re-evaluate cosine; it explains coverage.

    Returns:
      (kw_hit_rate, details) where details contains:
        - kw_hits_idx: indices of requirements passing the token-overlap gate
        - kw_misses_idx: indices failing the gate
        - token_intersections: list[(i, shared_count)]
    """
    def toks(s: str) -> set[str]:
        return set(re.findall(r"[\w\-/]+", s.lower()))

    details: Dict[str, List] = {"kw_hits_idx": [], "kw_misses_idx": [], "token_intersections": []}
    if not req_units or not feat_units or not best_match_indices:
        return 0.0, details
    kw_hits = 0
    for i, j in enumerate(best_match_indices):
        if j < 0 or j >= len(feat_units):
            details["kw_misses_idx"].append(i)
            details["token_intersections"].append((i, 0))
            continue
        inter = len(toks(req_units[i]) & toks(feat_units[j]))
        details["token_intersections"].append((i, inter))
        if inter >= min_shared_tokens:
            kw_hits += 1
            details["kw_hits_idx"].append(i)
        else:
            details["kw_misses_idx"].append(i)
    kw_hit_rate = kw_hits / max(1, len(best_match_indices))
    return kw_hit_rate, details


def get_description() -> str:
    return (
        "kw_hit_rate — доля требований, у которых с лучшим совпадением в шагах есть ≥2 общих токена. "
        "Это лексическая трассировка: шаги должны использовать ту же терминологию, что и .md."
    )


def get_detailed_fix() -> str:
    return (
        "Согласуйте лексику: перенесите ключевые термины из .md в Then/And (те же сущности/атрибуты/направления). "
        "Практика: для каждого непокрытого пункта возьмите best‑совпадение и добавьте в шаг 1–2 лексемы из требования до достижения ≥2 общих токенов. "
        "Избегайте синонимов и форм изменяющих корень; используйте те же ключевые слова, что в .md."
    )

