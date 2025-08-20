import os
import re
from typing import Any, Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer, util
from .nli_then import detect_then_contradictions_doc
from core.models import nli_contradiction_prob, nli_entailment_prob


def _token_set(s: str) -> set[str]:
    """Return a set of lowercase word-like tokens for quick overlap checks."""
    return set(re.findall(r"[\w\-/]+", s.lower()))


def normalize_subject_signature(text: str) -> str:
    """Normalize Then-like subject strings for approximate identity comparisons."""
    s = text.lower()
    s = re.sub(r"^(тогда|then|и|and)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b[0-9]{3}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_then_by_scenario(text: str) -> List[List[str]]:
    """Group Then/And lines per scenario as lists of statements."""
    scenarios: List[List[str]] = []
    current: List[str] = []
    in_then = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^(Сценарий:|Scenario:)\b", line, flags=re.IGNORECASE):
            if current:
                scenarios.append(current)
            current = []
            in_then = False
            continue
        if re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE):
            m = re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE)
            if m:
                current.append(m.group(2).strip())
                in_then = True
            continue
        if in_then:
            m_and = re.match(r"^(И|And)\s+(.+)$", line, re.IGNORECASE)
            if m_and:
                current.append(m_and.group(2).strip())
            else:
                in_then = False
    if current:
        scenarios.append(current)
    return scenarios


def detect_then_negation_contradictions(feat_text: str, model: SentenceTransformer) -> float:
    """Weight contradictions from near-duplicate Then statements with opposite polarity.

    Purpose:
      Catch pairs like "показывается" vs "не показывается" при прочих равных.

    Env tuning:
      - THEN_NEG_SIM: cosine similarity threshold between Then statements (default 0.80)
      - THEN_PAIR_JACC_MAX: maximum token Jaccard to consider as different surface (default 0.40)
      - THEN_NEG_REQUIRE_NEGATION: require explicit negation polarity difference (default 1)
    """
    scenarios = extract_then_by_scenario(feat_text)
    if not scenarios:
        return 0.0
    weight = 0.0
    for thens in scenarios:
        if not thens:
            continue
        emb = model.encode(thens, convert_to_tensor=True, normalize_embeddings=True)
        sim = util.cos_sim(emb, emb)
        require_negation = str(os.environ.get("THEN_NEG_REQUIRE_NEGATION", "1")).lower() in ("1", "true", "yes", "on")
        neg_tokens = set(["не", "no", "not", "без", "never", "cannot", "can't"]) if require_negation else set()
        for i in range(len(thens)):
            for j in range(i + 1, len(thens)):
                if float(sim[i, j].item()) < float(os.environ.get("THEN_NEG_SIM", "0.80")):
                    continue
                a = thens[i].lower(); b = thens[j].lower()
                inter = len(_token_set(a) & _token_set(b)); union = len(_token_set(a) | _token_set(b))
                jacc = inter / max(1, union)
                if jacc <= float(os.environ.get("THEN_PAIR_JACC_MAX", "0.40")) and a != b:
                    if require_negation:
                        has_neg_a = bool(neg_tokens & _token_set(a))
                        has_neg_b = bool(neg_tokens & _token_set(b))
                        if has_neg_a != has_neg_b:
                            weight += 1.0
                    else:
                        weight += 1.0
    return weight


def extract_gw_then_blocks(text: str) -> List[Dict[str, Any]]:
    """Extract per-scenario dicts: {title, gw, then[]} for GW-divergence/Title/Adj checks.

    gw: накопленные строки вне блока Then (последовательности Given/When/контекст)
    then: список строк после "Тогда" и следующих "И" до конца блока
    """
    blocks: List[Dict[str, Any]] = []
    current_gw: List[str] = []
    current_then: List[str] = []
    in_then = False
    current_title = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m_title = re.match(r"^(Сценарий:|Scenario:)\s*(.+)$", line, re.IGNORECASE)
        if m_title:
            if current_gw or current_then:
                blocks.append({"title": current_title, "gw": "\n".join(current_gw), "then": current_then[:]})
            current_title = m_title.group(2).strip()
            current_gw = []
            current_then = []
            in_then = False
            continue
        if re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE):
            m = re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE)
            if m:
                current_then.append(m.group(2).strip())
                in_then = True
            continue
        if in_then:
            m_and = re.match(r"^(И|And)\s+(.+)$", line, re.IGNORECASE)
            if m_and:
                current_then.append(m_and.group(2).strip())
            else:
                in_then = False
                current_gw.append(line)
        else:
            if not re.match(r"^(@|#)", line):
                current_gw.append(line)
    if current_gw or current_then:
        blocks.append({"title": current_title, "gw": "\n".join(current_gw), "then": current_then[:]})
    return blocks


def detect_same_setup_divergent_outcomes(feat_text: str, model: SentenceTransformer) -> float:
    """If Given/When are highly similar across scenarios but Then outcomes diverge, add weight (GW-div)."""
    blocks = extract_gw_then_blocks(feat_text)
    if len(blocks) < 2:
        return 0.0
    gw_texts = [b["gw"] for b in blocks]
    gw_emb = model.encode(gw_texts, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(gw_emb, gw_emb)
    weight = 0.0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if float(sim[i, j].item()) < float(os.environ.get("GW_SIM_THRESH", "0.88")):
                continue
            a_then = set(blocks[i]["then"]) ; b_then = set(blocks[j]["then"])
            if not a_then or not b_then:
                continue
            inter = len(a_then & b_then)
            union = len(a_then | b_then)
            jacc = inter / max(1, union)
            a_list = list(a_then); b_list = list(b_then)
            a_emb = model.encode(a_list, convert_to_tensor=True, normalize_embeddings=True)
            b_emb = model.encode(b_list, convert_to_tensor=True, normalize_embeddings=True)
            a_cent = a_emb.mean(dim=0, keepdim=True)
            b_cent = b_emb.mean(dim=0, keepdim=True)
            cent_sim = float(util.cos_sim(a_cent, b_cent).item())
            if jacc <= float(os.environ.get("THEN_JACC_MAX", "0.40")) and cent_sim <= float(os.environ.get("THEN_CENTROID_MAX", "0.70")):
                weight += 1.0
    return weight


def detect_similar_title_divergence(feat_text: str, model: SentenceTransformer) -> float:
    """Similar scenario titles with divergent outcomes -> contradiction weight (Title-div)."""
    blocks = extract_gw_then_blocks(feat_text)
    titles = [b.get("title", "") for b in blocks]
    idx = [i for i, t in enumerate(titles) if t]
    if len(idx) < 2:
        return 0.0
    texts = [titles[i] for i in idx]
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb, emb)
    weight = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if float(sim[a, b].item()) < float(os.environ.get("TITLE_SIM_MIN", "0.86")):
                continue
            a_then = set(blocks[idx[a]]["then"]) ; b_then = set(blocks[idx[b]]["then"])
            if not a_then or not b_then:
                continue
            inter = len(a_then & b_then); union = len(a_then | b_then)
            jacc = inter / max(1, union)
            if jacc <= float(os.environ.get("TITLE_THEN_JACC_MAX", "0.40")):
                weight += 1.0
    return weight


def detect_adjacent_divergence(feat_text: str, model: SentenceTransformer) -> float:
    """Adjacent scenarios with similar GW but divergent Then -> contradiction weight (Adj-div).

    Env tuning:
      - GW_SIM_ADJ: cosine threshold for adjacent GW similarity (default 0.86)
      - ADJ_THEN_JACC_MAX: max Jaccard overlap to consider outcomes divergent (default 0.40)
      - ADJ_REQUIRE_THEN_NLI: if set, require minimal NLI contradiction on enriched Then (default 0)
    """
    blocks = extract_gw_then_blocks(feat_text)
    if len(blocks) < 2:
        return 0.0
    gw = [b["gw"] for b in blocks]
    emb = model.encode(gw, convert_to_tensor=True, normalize_embeddings=True)
    weight = 0.0
    require_then_nli = str(os.environ.get("ADJ_REQUIRE_THEN_NLI", "0")).lower() in ("1", "true", "yes", "on")
    for i in range(len(blocks) - 1):
        j = i + 1
        sim_gw = float(util.cos_sim(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item())
        if sim_gw < float(os.environ.get("GW_SIM_ADJ", "0.86")):
            continue
        a_then = set(blocks[i]["then"]) ; b_then = set(blocks[j]["then"])
        if not a_then or not b_then:
            continue
        inter = len(a_then & b_then); union = len(a_then | b_then)
        jacc = inter / max(1, union)
        if jacc <= float(os.environ.get("ADJ_THEN_JACC_MAX", "0.40")):
            if not require_then_nli:
                weight += 1.0
            else:
                # Minimal NLI gate on concatenated outcomes
                try:
                    from core.models import nli_contradiction_prob, nli_entailment_prob
                    premise = "\n".join(sorted(list(a_then)))
                    hypothesis = "\n".join(sorted(list(b_then)))
                    c = nli_contradiction_prob(premise, hypothesis)
                    e = nli_entailment_prob(premise, hypothesis)
                    if (c >= float(os.environ.get("ADJ_NLI_CONTRA_MIN", "0.65"))) and ((c - e) >= float(os.environ.get("ADJ_NLI_MARGIN", "0.15"))):
                        weight += 1.0
                except Exception:
                    pass
    return weight


def _split_paragraphs_md(text: str) -> List[str]:
    """Split MD into logical paragraphs.
    Rules:
      - Headers (lines starting with '#') are ignored
      - List items ("- ", "* ", "• ") are treated as standalone paragraphs
      - Non-empty lines are aggregated until a blank line separates paragraphs
    """
    paragraphs: List[str] = []
    buf: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if buf:
                paragraphs.append(re.sub(r"\s+", " ", " ".join(buf)).strip())
                buf = []
            continue
        if line.startswith("#"):
            # flush buffer on header boundary
            if buf:
                paragraphs.append(re.sub(r"\s+", " ", " ".join(buf)).strip())
                buf = []
            continue
        # list item as own paragraph
        if re.match(r"^[-*•]\s+", line):
            if buf:
                paragraphs.append(re.sub(r"\s+", " ", " ".join(buf)).strip())
                buf = []
            paragraphs.append(re.sub(r"\s+", " ", re.sub(r"^[-*•]\s+", "", line)).strip())
            continue
        buf.append(line)
    if buf:
        paragraphs.append(re.sub(r"\s+", " ", " ".join(buf)).strip())
    # filter too-short paragraphs
    paragraphs = [p for p in paragraphs if len(p) >= 3]
    return paragraphs


def detect_md_internal_contradictions(md_text: str, model: SentenceTransformer) -> Tuple[float, float]:
    """Detect contradictions inside the requirements (.md) document.
    Approach:
      - Split MD text into requirement-like sentences
      - Encode and filter candidate pairs by semantic similarity and token overlap
      - Run NLI in both directions to confirm contradiction with margin over entailment
    Returns a weighted count of contradictory pairs.
    """
    req_units = _split_paragraphs_md(md_text)
    if len(req_units) < 2:
        return 0.0, 0.0

    emb = model.encode(req_units, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb, emb)

    sim_thresh = float(os.environ.get("MD_SIM_THRESH", "0.82"))
    nli_contra_thresh = float(os.environ.get("MD_NLI_CONTRA_THRESHOLD", "0.72"))
    nli_margin = float(os.environ.get("MD_NLI_MARGIN", "0.18"))
    min_shared_tokens = int(os.environ.get("MD_MIN_SHARED_TOKENS", "2"))

    neg_tokens = set(["не", "no", "not", "без", "never", "cannot", "can't"])

    nli_weight = 0.0
    total_weight = 0.0
    for i in range(len(req_units)):
        ai = req_units[i]
        for j in range(i + 1, len(req_units)):
            score = float(sim[i, j].item())
            if score < sim_thresh:
                continue
            aj = req_units[j]
            # quick token overlap gate
            inter = len(_token_set(ai) & _token_set(aj))
            if inter < min_shared_tokens:
                continue

            # NLI both directions with margin over entailment
            try:
                c1 = nli_contradiction_prob(ai, aj)
                e1 = nli_entailment_prob(ai, aj)
                c2 = nli_contradiction_prob(aj, ai)
                e2 = nli_entailment_prob(aj, ai)
                c = max(c1, c2)
                e = max(e1, e2)
            except Exception:
                c = 0.0
                e = 0.0

            if (c >= nli_contra_thresh) and ((c - e) >= nli_margin):
                add = 1.0 if c < 0.80 else 1.5
                nli_weight += add
                total_weight += add
                continue

            # Secondary heuristic: strong similarity + opposite negation polarity
            toks_i = _token_set(ai)
            toks_j = _token_set(aj)
            has_neg_i = bool(neg_tokens & toks_i)
            has_neg_j = bool(neg_tokens & toks_j)
            if has_neg_i != has_neg_j and score >= max(sim_thresh, 0.86):
                total_weight += 0.5

    return nli_weight, total_weight


def compute_contradictions(md_text: str, feat_text: str, model: SentenceTransformer) -> Tuple[bool, Dict[str, float]]:
    """Aggregate contradiction signals from feature tests and .md document.

    Returns (contra_flag, details) where flag is boolean and details contains component weights.
    """
    doc_then_contra = detect_then_contradictions_doc(feat_text, model)
    then_neg = detect_then_negation_contradictions(feat_text, model)
    gw_div = detect_same_setup_divergent_outcomes(feat_text, model)
    title_div = detect_similar_title_divergence(feat_text, model)
    adj_div = detect_adjacent_divergence(feat_text, model)

    # Build example pairs for explainability (non-blocking)
    gw_pairs: List[Dict[str, Any]] = []
    title_pairs: List[Dict[str, Any]] = []
    adj_pairs: List[Dict[str, Any]] = []
    try:
        blocks = extract_gw_then_blocks(feat_text)
        if blocks:
            gw_texts = [b["gw"] for b in blocks]
            gw_emb = model.encode(gw_texts, convert_to_tensor=True, normalize_embeddings=True)
            sim = util.cos_sim(gw_emb, gw_emb)
            gw_sim_thresh = float(os.environ.get("GW_SIM_THRESH", "0.88"))
            then_jacc_max = float(os.environ.get("THEN_JACC_MAX", "0.40"))
            then_centroid_max = float(os.environ.get("THEN_CENTROID_MAX", "0.70"))
            # Collect GW-div pairs
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    if float(sim[i, j].item()) < gw_sim_thresh:
                        continue
                    a_then = set(blocks[i]["then"]) ; b_then = set(blocks[j]["then"])
                    if not a_then or not b_then:
                        continue
                    inter = len(a_then & b_then); union = len(a_then | b_then)
                    jacc = inter / max(1, union)
                    a_list = list(a_then); b_list = list(b_then)
                    a_emb = model.encode(a_list, convert_to_tensor=True, normalize_embeddings=True)
                    b_emb = model.encode(b_list, convert_to_tensor=True, normalize_embeddings=True)
                    a_cent = a_emb.mean(dim=0, keepdim=True)
                    b_cent = b_emb.mean(dim=0, keepdim=True)
                    cent_sim = float(util.cos_sim(a_cent, b_cent).item())
                    if jacc <= then_jacc_max and cent_sim <= then_centroid_max:
                        gw_pairs.append({
                            "a_title": blocks[i].get("title", f"scenario[{i}]") or f"scenario[{i}]",
                            "b_title": blocks[j].get("title", f"scenario[{j}]") or f"scenario[{j}]",
                            "a_gw": blocks[i].get("gw", ""),
                            "b_gw": blocks[j].get("gw", ""),
                            "a_then": list(a_then)[:3],
                            "b_then": list(b_then)[:3],
                        })
            # Title-div pairs
            titles = [b.get("title", "") for b in blocks]
            idx = [k for k, t in enumerate(titles) if t]
            if len(idx) >= 2:
                texts = [titles[k] for k in idx]
                emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
                tsim = util.cos_sim(emb, emb)
                title_sim_min = float(os.environ.get("TITLE_SIM_MIN", "0.86"))
                title_then_jacc_max = float(os.environ.get("TITLE_THEN_JACC_MAX", "0.40"))
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        if float(tsim[a, b].item()) < title_sim_min:
                            continue
                        a_then = set(blocks[idx[a]]["then"]) ; b_then = set(blocks[idx[b]]["then"])
                        if not a_then or not b_then:
                            continue
                        inter = len(a_then & b_then); union = len(a_then | b_then)
                        jacc = inter / max(1, union)
                        if jacc <= title_then_jacc_max:
                            title_pairs.append({
                                "a_title": blocks[idx[a]].get("title", f"scenario[{idx[a]}]") or f"scenario[{idx[a]}]",
                                "b_title": blocks[idx[b]].get("title", f"scenario[{idx[b]}]") or f"scenario[{idx[b]}]",
                                "a_then": list(a_then)[:3],
                                "b_then": list(b_then)[:3],
                            })
            # Adjacent-div pairs
            gw = [b["gw"] for b in blocks]
            emb2 = model.encode(gw, convert_to_tensor=True, normalize_embeddings=True)
            gw_sim_adj = float(os.environ.get("GW_SIM_ADJ", "0.86"))
            adj_jacc_max = float(os.environ.get("ADJ_THEN_JACC_MAX", "0.40"))
            for i in range(len(blocks) - 1):
                j = i + 1
                sim_gw = float(util.cos_sim(emb2[i].unsqueeze(0), emb2[j].unsqueeze(0)).item())
                if sim_gw < gw_sim_adj:
                    continue
                a_then = set(blocks[i]["then"]) ; b_then = set(blocks[j]["then"])
                if not a_then or not b_then:
                    continue
                inter = len(a_then & b_then); union = len(a_then | b_then)
                jacc = inter / max(1, union)
                if jacc <= adj_jacc_max:
                    adj_pairs.append({
                        "a_title": blocks[i].get("title", f"scenario[{i}]") or f"scenario[{i}]",
                        "b_title": blocks[j].get("title", f"scenario[{j}]") or f"scenario[{j}]",
                        "a_gw": blocks[i].get("gw", ""),
                        "b_gw": blocks[j].get("gw", ""),
                        "a_then": list(a_then)[:3],
                        "b_then": list(b_then)[:3],
                    })
    except Exception:
        gw_pairs = []
        title_pairs = []
        adj_pairs = []
    md_nli_contra, md_contra = detect_md_internal_contradictions(md_text, model)
    md_enforce = str(os.environ.get("MD_CONTRA_ENFORCE", "0")).lower() in ("1", "true", "yes", "on")
    # Simple aggregation policy: strong NLI OR any two structural signals
    contra = False
    # Slightly increase strong NLI threshold to reflect more conservative ensemble
    if doc_then_contra >= float(os.environ.get("THEN_CONTRA_STRONG", "3.2")):
        contra = True
    else:
        signals = 0
        signals += 1 if then_neg >= float(os.environ.get("THEN_NEG_CONTRA_MIN", "2.2")) else 0
        signals += 1 if gw_div >= float(os.environ.get("GW_DIV_MIN", "1.2")) else 0
        signals += 1 if title_div >= float(os.environ.get("TITLE_DIV_MIN", "1.2")) else 0
        signals += 1 if adj_div >= float(os.environ.get("ADJ_DIV_MIN", "1.2")) else 0
        # MD contradictions: by default informational only; enable impact via MD_CONTRA_ENFORCE
        if md_enforce:
            md_strong = float(os.environ.get("MD_CONTRA_STRONG", "2.0"))
            md_min_signal = float(os.environ.get("MD_CONTRA_MIN_SIGNAL", "1.0"))
            if md_contra >= md_strong:
                contra = True
            else:
                signals += 1 if md_contra >= md_min_signal else 0
        if signals >= int(os.environ.get("CONTRA_MIN_SIGNALS", "2")):
            contra = True
    details = {
        "doc_then_contra": doc_then_contra,
        "then_neg": then_neg,
        "gw_div": gw_div,
        "title_div": title_div,
        "adj_div": adj_div,
        "feature_contradiction": md_contra,
        "md_nli_contra": md_nli_contra,
        # Explainability payloads
        "gw_pairs": gw_pairs[:3],
        "title_pairs": title_pairs[:3],
        "adj_pairs": adj_pairs[:3],
    }
    return contra, details


def get_description() -> str:
    return (
        "contradictions — признаки структурных/NLI‑противоречий: Then‑NLI, Then‑NEG, GW/Title/Adj‑div, а также внутренние противоречия в .md."
    )


def get_detailed_fix() -> str:
    return (
        "Уберите противоположные Then при одинаковых условиях; различайте предусловия Given/When; "
        "объединяйте дублирующиеся сценарии или переводите в Scenario Outline; согласуйте ‘Функция:’ с ‘Цель:’."
    )

