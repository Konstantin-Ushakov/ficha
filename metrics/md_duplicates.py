import os
import re
from typing import Dict, List, Tuple

from .extras import _word_tokens, _bm25_best, _tfidf_cosine_best


def _normalize(text: str) -> str:
    # drop headings and bold markers for noise reduction
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        lines.append(s)
    return "\n".join(lines)


def compute_md_duplicates(md_items: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, float, float]]]:
    """Compute duplicates across MD files.

    Args:
      md_items: list of (name, acceptance_text) pairs

    Returns:
      Map name -> list of (other_name, bm25, tfidf) sorted by combined score desc.
    """
    # Prefer duplicate-specific toggles, fallback to generic IR toggles
    use_bm25 = str(os.environ.get("DUPS_BM25_ENABLE", os.environ.get("BM25_ENABLE", "1"))).lower() in ("1", "true", "yes", "on")
    use_tfidf = str(os.environ.get("DUPS_TFIDF_ENABLE", os.environ.get("TFIDF_ENABLE", "1"))).lower() in ("1", "true", "yes", "on")
    if not (use_bm25 or use_tfidf):
        return {}

    # Tokenize documents once
    docs_tokens = [(name, [_word_tokens(line) for line in _normalize(text).splitlines() if line.strip()]) for name, text in md_items]
    result: Dict[str, List[Tuple[str, float, float]]] = {}
    for i, (name_i, toks_i) in enumerate(docs_tokens):
        try:
            from .util_log import log_once
            if use_bm25:
                log_once("md_duplicates_bm25")
            if use_tfidf:
                log_once("md_duplicates_tfidf")
        except Exception:
            pass
        # Join tokens back into strings for query formation
        q_lines = [" ".join(t) for t in toks_i] or [""]
        bests: List[Tuple[str, float, float]] = []
        for j, (name_j, toks_j) in enumerate(docs_tokens):
            if i == j:
                continue
            # treat doc i as a set of queries against doc j tokens
            bm25_total = 0.0
            tfidf_total = 0.0
            if use_bm25:
                for q in q_lines:
                    bm25_total += _bm25_best(q, toks_j)
            if use_tfidf:
                for q in q_lines:
                    tfidf_total += _tfidf_cosine_best(q, toks_j)
            # normalize by number of query lines
            n = max(1, len(q_lines))
            bests.append((name_j, bm25_total / n if n else 0.0, tfidf_total / n if n else 0.0))
        # sort by sum of enabled scores
        def key_fn(t: Tuple[str, float, float]) -> float:
            s = 0.0
            if use_bm25:
                s += t[1]
            if use_tfidf:
                s += t[2]
            return s
        bests.sort(key=lambda x: key_fn(x), reverse=True)
        topk = int(os.environ.get("MD_DUP_TOPK", "3"))
        result[name_i] = bests[:topk]
    return result


