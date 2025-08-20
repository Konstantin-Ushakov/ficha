import os
import re
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
from .hit_rate import compute_hit_rate
from .kw_hit_rate import compute_kw_hit_rate
from .nli_hit_rate import compute_nli_hit_rate
from .step_hit_rate import compute_step_hit_rate


def _split_sentences(text: str) -> List[str]:
    """Split free-form text into sentence-like units.

    Purpose:
      Normalize arbitrary text into short, comparable units for embedding matching.

    Behavior:
      - Trims headers starting with '#'
      - Splits by punctuation boundaries . ! ?
      - Removes leading bullet/marker characters
    """
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("# ").strip()
        for s in re.split(r"(?<=[.!?])\s+", line):
            s = s.strip(" -•\t")
            if len(s) >= 3:
                parts.append(s)
    return parts


def _extract_goal_and_body(md_text: str) -> Tuple[str, str]:
    """Extract single-line goal (line starting with 'Цель:') and return remaining body.

    Returns:
      (goal_text, body_without_goal_line)

    Notes:
      The goal line is used exclusively by goal alignment (see compute_goal_alignment).
    """
    goal_text = ""
    body_lines: List[str] = []
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            body_lines.append(raw)
            continue
        # Support both plain 'Цель:' and bold markdown '**Цель:**'
        m = re.match(r"^(?:\*\*)?Цель:(?:\*\*)?\s*(.+)$", line)
        if m and not goal_text:
            goal_text = m.group(1).strip()
            # skip adding this line to body
            continue
        body_lines.append(raw)
    return goal_text, "\n".join(body_lines)


def _split_requirements(md_text: str) -> Tuple[List[str], str]:
    goal, body = _extract_goal_and_body(md_text)
    return _split_sentences(body), goal


def _extract_acceptance_section(md_text: str) -> Tuple[str, str]:
    """Return (acceptance_text, goal_text).

    Parser rules:
      - Goal is the first line starting with 'Цель:' (used elsewhere).
      - Acceptance section is the content under the heading starting with
        '## Описание фичи' up to the next '## ' heading or end of document.
      - If section is missing, fallback to the whole body (minus goal line).
    """
    goal_text, body = _extract_goal_and_body(md_text)
    # Prefer new parser behavior: section '## Описание фичи'
    lines = md_text.splitlines()
    start_idx = None
    end_idx = None
    for i, raw in enumerate(lines):
        if re.match(r"^\s*##\s+Критерии\s+приемки\b", raw.strip(), flags=re.IGNORECASE):
            start_idx = i + 1
            break
    if start_idx is not None:
        for j in range(start_idx, len(lines)):
            if re.match(r"^\s*##\s+", lines[j].strip()):
                end_idx = j
                break
        acc_text = "\n".join(lines[start_idx:end_idx]) if end_idx is not None else "\n".join(lines[start_idx:])
        return acc_text, goal_text
    # Fallback to previous behavior: body without goal
    return body, goal_text


def _split_semantic_requirements(md_text: str) -> Tuple[List[str], str]:
    """Split acceptance section of `.md` into semantic requirement units ("thoughts").

    Algorithm:
      - Extract 'Цель:' and isolate only the '## Описание фичи' section
      - Split into paragraphs; treat bullet/numbered lists as separate units
      - For non-list paragraphs: split into sentences, then further split long sentences by ';' and simple conjunctions

    Rationale:
      Используем только проверяемые, наблюдаемые исходы из секции «Описание фичи»,
      чтобы повысить точность покрытия Then/And.
    """
    body, goal = _extract_acceptance_section(md_text)
    units: List[str] = []
    paragraphs = re.split(r"\n\s*\n", body)
    for par in paragraphs:
        if not par or not par.strip():
            continue
        lines = [ln.strip() for ln in par.splitlines() if ln.strip()]
        if not lines:
            continue
        is_list = any(re.match(r"^(\*|-|•|\d+[\.)])\s+", ln) for ln in lines)
        if is_list:
            for ln in lines:
                ln_clean = re.sub(r"^(\*|-|•|\d+[\.)])\s+", "", ln).strip()
                if not ln_clean:
                    continue
                for s in _split_sentences(ln_clean):
                    if len(s) >= 3:
                        units.append(s)
            continue
        sentences = _split_sentences(par)
        for s in sentences:
            parts = re.split(r"\s*;\s*|\s+(?:и|или|and|or)\s+", s, flags=re.IGNORECASE)
            subparts = [p.strip(" -•\t") for p in parts if len(p.strip()) >= 8]
            if len(subparts) >= 2:
                units.extend(subparts)
            else:
                units.append(s)
    return units, goal


def _split_feature_units(feat_text: str) -> List[str]:
    """Extract feature-test textual units for semantic matching.

    - Drops tags and comments
    - Strips leading Gherkin keywords (Given/When/Then/And, локализованные аналоги)
    - Splits remaining lines into sentences
    """
    lines: List[str] = []
    for raw in feat_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if raw.lstrip()[:1] in ("#", "@"):
            continue
        line = re.sub(r"^(Функция:|Сценарий:|Предыстория:|Given|When|Then|And|But|Допустим|Когда|Тогда|И)\s*",
                      "", line, flags=re.IGNORECASE)
        if len(line) >= 3:
            lines.append(line)
    units: List[str] = []
    for l in lines:
        units.extend(_split_sentences(l))
    return units


def _extract_feature_descriptions(feat_text: str) -> List[str]:
    """Return all 'Функция:' descriptions from one or more feature files.

    These lines are used to compute goal alignment with the single-line 'Цель:' from `.md`.
    """
    descs: List[str] = []
    for raw in feat_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(Функция:|Feature:)\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            descs.append(m.group(2).strip())
    return descs


def compute_coverage(md_text: str, feat_text: str, model: SentenceTransformer) -> Dict[str, float]:
    """Compute coverage metrics between acceptance requirements (.md) and `.feature` steps.

    Definitions (see also individual metric files under tests/analyze/metrics):
      - hit_rate: fraction of acceptance units covered by any feature unit with cos ≥ SIM_THRESHOLD.
      - kw_hit_rate: fraction of acceptance units whose best match shares ≥2 alphanumeric tokens (keyword-overlap gate).
      - nli_hit_rate: NLI-based entailment coverage of Then/And vs acceptance units.
      - step_hit_rate: step-level structural coverage by keyword overlap vs Then/And.

    semantic_coverage in evaluation:
      Passes if any of (kw_hit_rate ≥ 0.60) OR (nli_hit_rate ≥ 0.60) OR (step_hit_rate ≥ 0.60);
      when agg_base ≥ 0.82, allow weaker KW path: kw_hit_rate ≥ 0.40.
    """
    sim_threshold = float(os.environ.get("SIM_THRESHOLD", "0.50"))
    # Log metric usage once
    try:
        from .util_log import log_once
        if str(os.environ.get("HIT_ENABLE", "1")).lower() in ("1","true","yes","on"):
            log_once("hit_rate")
        if str(os.environ.get("KW_HIT_ENABLE", "1")).lower() in ("1","true","yes","on"):
            log_once("kw_hit_rate")
        if str(os.environ.get("NLI_HIT_ENABLE", "0")).lower() in ("1","true","yes","on"):
            log_once("nli_hit_rate")
        if str(os.environ.get("STEP_HIT_ENABLE", "1")).lower() in ("1","true","yes","on"):
            log_once("step_hit_rate")
    except Exception:
        pass
    # Feature toggles (env exported from config.metrics)
    enable_hit = str(os.environ.get("HIT_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    enable_kw = str(os.environ.get("KW_HIT_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    enable_nli = str(os.environ.get("NLI_HIT_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    enable_step = str(os.environ.get("STEP_HIT_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    # Use semantic unit splitter (thought-level) for MD requirements
    req_units, _goal_text = _split_semantic_requirements(md_text)
    feat_units_all = _split_feature_units(feat_text)
    hit_rate = 0.0
    kw_hit_rate = 0.0
    entailment_hit_rate = 0.0
    step_hit_rate = 0.0
    # Debug/explainability payloads
    best_idx: List[int] = []
    misses_idx: List[int] = []
    kw_misses_idx: List[int] = []
    nli_not_entailed_idx: List[int] = []
    step_misses_idx: List[int] = []
    if req_units and feat_units_all:
        # 1) hit_rate + best-match indices for KW explanation
        if enable_hit:
            hr, hr_details = compute_hit_rate(req_units, feat_units_all, model, sim_threshold)
            hit_rate = hr
        else:
            hr_details = {"matches": [], "misses_idx": []}
        # Build best-match arrays aligned with requirements
        best_sims: List[float] = []
        best_texts: List[str] = []
        if hr_details.get("matches"):
            # matches have one tuple per requirement row in order
            best_idx = [int(j) for (_, j, _sim) in hr_details["matches"]]
            best_sims = [float(sim) for (_i, _j, sim) in hr_details["matches"]]
            best_texts = [feat_units_all[j] if 0 <= j < len(feat_units_all) else "" for j in best_idx]
        misses_idx = list(hr_details.get("misses_idx", []))
        # 2) kw_hit_rate explanation on best matches
        if enable_kw:
            kw, kw_details = compute_kw_hit_rate(req_units, feat_units_all, best_idx, min_shared_tokens=2)
            kw_hit_rate = kw
        else:
            kw_details = {"kw_misses_idx": []}
        kw_misses_idx = list(kw_details.get("kw_misses_idx", []))
        # 3) nli_hit_rate on Then/And statements (no GW context here)
        nli_entail_threshold = float(os.environ.get("NLI_HIT_ENTAIL", "0.70"))
        if enable_nli:
            entailment_hit_rate, nli_details = compute_nli_hit_rate(
                req_units,
                feat_text,
                model,
                entailment_threshold=nli_entail_threshold,
            )
            nli_not_entailed_idx = list(nli_details.get("not_entailed_idx", []))
        else:
            nli_details = {"best_pairs": [], "best_top3_pairs": [], "worst_pairs": [], "not_entailed_idx": []}
        # 4) step_hit_rate on Then/And keyword overlap
        if enable_step:
            step_hit_rate, step_details = compute_step_hit_rate(req_units, feat_text, min_keyword_overlap=2)
            step_misses_idx = list(step_details.get("step_misses_idx", []))
        else:
            step_details = {"step_misses_idx": [], "step_best_then": [], "step_token_overlap": []}
    return {
        "hit_rate": hit_rate,
        "kw_hit_rate": kw_hit_rate,
        "nli_hit_rate": entailment_hit_rate,
        "step_hit_rate": step_hit_rate,
        # Debug fields for explainability in CLI
        "req_units": req_units,
        "best_match_indices": best_idx,
        "best_match_sims": best_sims if 'best_sims' in locals() else [],
        "best_match_texts": best_texts if 'best_texts' in locals() else [],
        "misses_idx": misses_idx,
        "kw_misses_idx": kw_misses_idx,
        "nli_not_entailed_idx": nli_not_entailed_idx,
        # NLI explainability payload
        "nli_best_pairs": nli_details.get("best_pairs", []) if 'nli_details' in locals() else [],
        "nli_best_top3_pairs": nli_details.get("best_top3_pairs", []) if 'nli_details' in locals() else [],
        "nli_worst_pairs": nli_details.get("worst_pairs", []) if 'nli_details' in locals() else [],
        "nli_entail_threshold": nli_entail_threshold if (req_units and feat_units_all) else float(os.environ.get("NLI_HIT_ENTAIL", "0.70")),
        "step_misses_idx": step_misses_idx,
        "step_best_then": step_details.get("step_best_then", []) if 'step_details' in locals() else [],
        "step_token_overlap": step_details.get("step_token_overlap", []) if 'step_details' in locals() else [],
    }


def compute_goal_alignment(md_text: str, feat_text: str, model: SentenceTransformer) -> Dict[str, float | bool]:
    """Compute alignment between 'Цель:' (from `.md`) and 'Функция:' (from `.feature`).

    Output:
      - goal_sim ∈ [0,1]: cosine similarity between the goal line and the concatenation of all feature descriptions.
      - goal_hit: boolean (goal_sim ≥ GOAL_SIM_THRESHOLD, default 0.60).

    How to improve goal_sim in practice:
      - Keep 'Функция:' semantically close to 'Цель:' (same terminology, same user value phrasing).
      - Avoid technical jargon in 'Функция:' that is absent in 'Цель:'.
    """
    _req_units, goal_text = _split_requirements(md_text)
    feat_descs = _extract_feature_descriptions(feat_text)
    goal_sim = 0.0
    goal_hit = False
    worst_pairs: List[Tuple[str, float]] = []
    if goal_text and feat_descs:
        concatenated_desc = " \n".join(feat_descs)
        # Main goal similarity vs all descriptions concatenated
        emb = model.encode([goal_text, concatenated_desc], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(emb[0].unsqueeze(0), emb[1].unsqueeze(0)).item())
        goal_sim = sim
        goal_hit = goal_sim >= float(os.environ.get("GOAL_SIM_THRESHOLD", "0.60"))
        # Also compute per-description similarities to find the least aligned ones (bottom-3)
        try:
            cand_emb = model.encode([goal_text] + feat_descs, convert_to_tensor=True, normalize_embeddings=True)
            sims: List[Tuple[str, float]] = []
            for i, desc in enumerate(feat_descs):
                s = float(util.cos_sim(cand_emb[0].unsqueeze(0), cand_emb[i+1].unsqueeze(0)).item())
                sims.append((desc, s))
            sims.sort(key=lambda x: x[1])
            worst_pairs = sims[: min(3, len(sims))]
        except Exception:
            worst_pairs = []
    return {
        "goal_sim": goal_sim,
        "goal_hit": goal_hit,
        "goal_worst": worst_pairs,
    }


