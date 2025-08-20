import os
import re
from typing import Dict, List, Tuple


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[\w\-]+", text.lower(), flags=re.UNICODE)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _extract_scenarios_and_then(feat_text: str) -> List[Tuple[str, List[str]]]:
    scenarios: List[Tuple[str, List[str]]] = []
    current_title = None
    current_then: List[str] = []
    for raw in feat_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("@"):
            continue
        m = re.match(r"^(Сценарий:|Scenario:)\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            if current_title is not None:
                scenarios.append((current_title, current_then))
            current_title = m.group(2).strip()
            current_then = []
            continue
        if re.match(r"^(Тогда|And|Then|И)\s+", line, flags=re.IGNORECASE):
            current_then.append(re.sub(r"^(Тогда|And|Then|И)\s+", "", line, flags=re.IGNORECASE))
    if current_title is not None:
        scenarios.append((current_title, current_then))
    return scenarios


def _numbers(text: str) -> List[str]:
    return re.findall(r"\d+", text)


def _bm25_best(req: str, docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> float:
    # Minimal BM25 implementation over tokenized docs
    q = _word_tokens(req)
    if not q or not docs:
        return 0.0
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, N)
    # document frequencies
    df: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    # precompute IDF
    idf: Dict[str, float] = {}
    for t in set(q):
        n = df.get(t, 0)
        # using BM25+ like idf smoothing
        idf[t] = max(0.0, ( (N - n + 0.5) / (n + 0.5) ))
    best = 0.0
    for d in docs:
        score = 0.0
        dl = len(d) or 1
        tf: Dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        for t in q:
            if t not in idf:
                continue
            f = tf.get(t, 0)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += idf[t] * ((f * (k1 + 1)) / (denom or 1))
        if score > best:
            best = score
    return best


def _tfidf_cosine_best(req: str, docs: List[List[str]]) -> float:
    # Lightweight TF-IDF cosine without external deps
    q = _word_tokens(req)
    if not q or not docs:
        return 0.0
    vocab: Dict[str, int] = {}
    for t in set(q):
        vocab.setdefault(t, len(vocab))
    for d in docs:
        for t in set(d):
            vocab.setdefault(t, len(vocab))
    N = len(docs)
    # df
    df: Dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    # idf
    import math
    idf: Dict[str, float] = {t: math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0 for t in vocab}
    # query vector
    from collections import Counter
    q_tf = Counter(q)
    q_vec = {t: (q_tf.get(t, 0) * idf.get(t, 0.0)) for t in vocab}
    def _cos(doc: List[str]) -> float:
        tf = Counter(doc)
        d_vec = {t: (tf.get(t, 0) * idf.get(t, 0.0)) for t in vocab}
        # cosine
        num = sum(q_vec[t] * d_vec[t] for t in vocab)
        qn = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        dn = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        return num / (qn * dn)
    return max((_cos(d) for d in docs), default=0.0)


def compute_extras(md_text: str, feat_text: str, cov: Dict[str, object]) -> Dict[str, float]:
    # Toggles
    on = lambda k: str(os.environ.get(k, "0")).lower() in ("1", "true", "yes", "on")
    req_units: List[str] = cov.get("req_units", []) if isinstance(cov, dict) else []
    best_idx: List[int] = cov.get("best_match_indices", []) if isinstance(cov, dict) else []
    feat_best_texts: List[str] = cov.get("best_match_texts", []) if isinstance(cov, dict) else []
    # Tokenized feat units for IR metrics
    feat_units_tokens = [ _word_tokens(t) for t in (cov.get("best_match_texts", []) or []) ]
    # If best_texts are empty, fall back to splitting feat_text by lines
    if not feat_units_tokens:
        feat_units_tokens = [ _word_tokens(l) for l in feat_text.splitlines() if l.strip() ]

    out: Dict[str, float] = {}

    # MD structural sections check (feature-description compliance)
    # Require: title '# Фича:', Role/Goal/Value lines, '## Описание фичи', '## Связь с файлами' with '### Реализация' and at least one .feature path
    try:
        title_ok = bool(re.search(r"^\s*#\s*Фича\s*:\s*.+$", md_text, flags=re.MULTILINE))
        role_ok = bool(re.search(r"^\s*\*\*Роль:\*\*\s*.+$", md_text, flags=re.MULTILINE))
        goal_ok = bool(re.search(r"^\s*\*\*Цель:\*\*\s*.+$|^\s*Цель:\s*.+$", md_text, flags=re.MULTILINE))
        value_ok = bool(re.search(r"^\s*\*\*Ценность:\*\*\s*.+$", md_text, flags=re.MULTILINE))
        desc_ok = bool(re.search(r"^\s*##\s*Описание\s+фичи\s*$", md_text, flags=re.MULTILINE))
        link_ok = bool(re.search(r"^\s*##\s*Связь\s+с\s+файлами\s*$", md_text, flags=re.MULTILINE))
        impl_ok = bool(re.search(r"^\s*###\s*Реализация\s*$", md_text, flags=re.MULTILINE))
        # At least one .feature bullet after Реализация until next heading
        impl_block = re.search(r"###\s*Реализация\s*\n(.*?)(?=\n##|\n###|\Z)", md_text, flags=re.DOTALL)
        has_feature_path = False
        if impl_block:
            for line in impl_block.group(1).splitlines():
                if re.search(r"\btests/.*?\.feature\b", line.strip()):
                    has_feature_path = True
                    break
        md_sections = 1.0 if (title_ok and role_ok and goal_ok and value_ok and desc_ok and link_ok and impl_ok and has_feature_path) else 0.0
        out["md_sections"] = md_sections
    except Exception:
        out["md_sections"] = 0.0

    # Jaccard lemma overlap (approx by lowercase tokens)
    if on("JACCARD_LEMMA_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("jaccard_lemma_overlap")
        except Exception:
            pass
        total = 0.0
        for i, req in enumerate(req_units):
            req_t = _word_tokens(req)
            bt = feat_best_texts[i] if i < len(feat_best_texts) else ""
            then_t = _word_tokens(bt)
            total += _jaccard(req_t, then_t)
        out["jaccard_lemma_overlap"] = total / max(1, len(req_units))

    # Cardinality alignment: numeric tokens match
    if on("CARDINALITY_ALIGNMENT_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("cardinality_alignment")
        except Exception:
            pass
        ok = 0
        for i, req in enumerate(req_units):
            nums = _numbers(req)
            bt = feat_best_texts[i] if i < len(feat_best_texts) else ""
            nums_b = _numbers(bt)
            if not nums:
                ok += 1  # no numeric expectation → neutral success
            else:
                ok += 1 if nums and nums_b and set(nums) == set(nums_b) else 0
        out["cardinality_alignment"] = ok / max(1, len(req_units))

    # BM25Okapi coverage-style: average best BM25 over feat units
    if on("BM25_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("bm25okapi")
        except Exception:
            pass
        docs = [ _word_tokens(t) for t in (cov.get("best_match_texts", []) or []) ]
        if not docs:
            docs = [ _word_tokens(l) for l in feat_text.splitlines() if l.strip() ]
        total = 0.0
        for r in req_units:
            total += _bm25_best(r, docs)
        out["bm25okapi"] = total / max(1, len(req_units))

    # TF-IDF cosine: average best
    if on("TFIDF_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("tfidf_cosine")
        except Exception:
            pass
        docs = [ _word_tokens(t) for t in (cov.get("best_match_texts", []) or []) ]
        if not docs:
            docs = [ _word_tokens(l) for l in feat_text.splitlines() if l.strip() ]
        total = 0.0
        for r in req_units:
            total += _tfidf_cosine_best(r, docs)
        out["tfidf_cosine"] = total / max(1, len(req_units))

    # Graph coverage analysis: uniqueness of traces vs requirements
    if on("GRAPH_COVERAGE_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("graph_coverage")
        except Exception:
            pass
        uniq = len(set(best_idx)) if best_idx else 0
        out["graph_coverage"] = uniq / max(1, len(req_units))

    # Per-scenario correspondence: fraction of scenarios with at least one Then overlapping with any requirement token
    if on("PER_SCENARIO_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("per_scenario_alignment")
        except Exception:
            pass
        scenarios = _extract_scenarios_and_then(feat_text)
        if scenarios:
            req_tok = [set(_word_tokens(r)) for r in req_units]
            covered = 0
            for _title, thens in scenarios:
                th_ok = False
                for th in thens:
                    tset = set(_word_tokens(th))
                    # consider covered if any requirement shares ≥2 tokens
                    if any(len(tset & rt) >= 2 for rt in req_tok):
                        th_ok = True
                        break
                covered += 1 if th_ok else 0
            out["per_scenario_alignment"] = covered / max(1, len(scenarios))

    # Trace density: unique Then hits per requirement count (capped at 1)
    if on("TRACE_DENSITY_ENABLE") and req_units:
        try:
            from .util_log import log_once
            log_once("trace_density")
        except Exception:
            pass
        then_count = 0
        for raw in feat_text.splitlines():
            if re.match(r"^(Тогда|And|Then|И)\s+", raw.strip(), flags=re.IGNORECASE):
                then_count += 1
        out["trace_density"] = min(1.0, then_count / max(1, len(req_units)))

    # Feature header validity (second line must have @<name> and @<category>)
    if on("FEATURE_HEADER_VALID_ENABLE"):
        try:
            from .util_log import log_once
            log_once("feature_header_valid")
        except Exception:
            pass
        lines = [l.rstrip("\n") for l in feat_text.splitlines() if l.strip()]
        valid = 0.0
        if len(lines) >= 2 and lines[1].startswith("@"):
            tags = lines[1].split()
            has_cat = any(re.match(r"^@\d+$", t) for t in tags)
            has_name = len(tags) >= 2
            valid = 1.0 if (has_cat and has_name) else 0.0
        out["feature_header_valid"] = valid

    # Duplicate scenarios ratio: fraction of scenario title pairs with high Jaccard
    if on("DUP_SCENARIOS_ENABLE"):
        try:
            from .util_log import log_once
            log_once("duplicate_scenarios_ratio")
        except Exception:
            pass
        sc = _extract_scenarios_and_then(feat_text)
        dup = 0
        total_pairs = 0
        for i in range(len(sc)):
            for j in range(i + 1, len(sc)):
                total_pairs += 1
                ji = _jaccard(_word_tokens(sc[i][0]), _word_tokens(sc[j][0]))
                if ji >= float(os.environ.get("DUP_SCENARIO_JACC", "0.8")):
                    dup += 1
        out["duplicate_scenarios_ratio"] = (dup / total_pairs) if total_pairs > 0 else 0.0

    # Terminology consistency (approx by KW hit)
    if on("TERMINOLOGY_CONSISTENCY_ENABLE"):
        try:
            from .util_log import log_once
            log_once("terminology_consistency")
        except Exception:
            pass
        out["terminology_consistency"] = float(cov.get("kw_hit_rate", 0.0))

    # FVI_mit: structural validity of .md (goal + acceptance present)
    if on("FVI_MIT_ENABLE"):
        try:
            from .util_log import log_once
            log_once("fvi_mit")
        except Exception:
            pass
        goal_ok = bool(re.search(r"^\s*\*\*?Цель:|^\s*Цель:\s+", md_text, flags=re.MULTILINE))
        acc_ok = bool(re.search(r"^\s*##\s+Критерии\s+приемки\b", md_text, flags=re.MULTILINE | re.IGNORECASE))
        # heuristic: require ≥1 non-empty paragraph in acceptance
        acc_text_match = re.search(r"##\s*Критерии\s*приемки\s*\n(.*?)(?=\n##|\Z)", md_text, re.DOTALL | re.IGNORECASE)
        acc_len_ok = False
        if acc_text_match:
            pts = [p.strip() for p in re.split(r"\n\s*\n", acc_text_match.group(1)) if p.strip()]
            acc_len_ok = len(pts) >= 1
        out["fvi_mit"] = 1.0 if (goal_ok and acc_ok and acc_len_ok) else (0.5 if (goal_ok or acc_ok) else 0.0)

    return out


