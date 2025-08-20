import os
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from core.models import nli_entailment_prob, nli_contradiction_prob


def _token_set(s: str) -> set[str]:
    """Tokenize text into lowercase word-like tokens for overlap filters."""
    return set(re.findall(r"[\w\-/]+", s.lower()))


def extract_then_statements(text: str) -> List[str]:
    """Collect all Then/And statements across scenarios (raw strings, no context).

    Used for fast pair prefiltering before NLI checks with GW context.
    """
    lines = text.splitlines()
    out: List[str] = []
    in_then = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^(Функция:|Сценарий:|Предыстория:|Given|When|But|Допустим|Когда|Но)\b", line, re.IGNORECASE):
            in_then = False
        m_then = re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE)
        if m_then:
            out.append(m_then.group(2).strip())
            in_then = True
            continue
        if in_then:
            m_and = re.match(r"^(И|And)\s+(.+)$", line, re.IGNORECASE)
            if m_and:
                out.append(m_and.group(2).strip())
            else:
                in_then = False
    return out


def detect_then_contradictions_doc(feature_text: str, model: SentenceTransformer) -> float:
    """Contradiction weight using Then statements with Given/When (GW) context.
    Pipeline:
      1) Extract per-scenario blocks with accumulated GW and Then lines.
      2) Build enriched statements: "IF <GW>\nTHEN <Then>" (or just THEN when no GW).
      3) Pairwise prefilter on raw Then: cosine ≥ INTRA_PAIR_SIM_THRESH and ≥ INTRA_MIN_SHARED_TOKENS shared tokens.
      4) NLI on enriched pairs. Accept if contradiction ≥ NLI_CONTRA_THRESHOLD and (contradiction - entailment) ≥ INTRA_NLI_MARGIN.
      5) Optional second-stage NLI on raw Then (THEN_TWO_STAGE=1) to reduce false positives.
    """
    blocks = _extract_gw_then_blocks_with_context(feature_text)
    if not blocks:
        return 0.0

    raw_then: List[str] = []
    enriched_then: List[str] = []
    origin_block_idx: List[int] = []
    for bi, b in enumerate(blocks):
        gw_text = (b.get("gw") or "").strip()
        for t in b.get("then") or []:
            raw_then.append(t)
            if gw_text:
                enriched_then.append(f"IF {gw_text}\nTHEN {t}")
            else:
                enriched_then.append(f"THEN {t}")
            origin_block_idx.append(bi)

    if not raw_then:
        return 0.0

    contradiction_threshold = float(os.environ.get("NLI_CONTRA_THRESHOLD", "0.70"))
    similarity_threshold = float(os.environ.get("INTRA_PAIR_SIM_THRESH", "0.80"))
    nli_margin = float(os.environ.get("INTRA_NLI_MARGIN", "0.22"))
    min_token_overlap = int(os.environ.get("INTRA_MIN_SHARED_TOKENS", "2"))
    contradiction_weight_sum = 0.0

    # Pair filtering using only raw Then statements (fast)
    statement_embeddings = model.encode(raw_then, convert_to_tensor=True, normalize_embeddings=True)
    similarity_matrix = util.cos_sim(statement_embeddings, statement_embeddings)

    # Compute GW similarity between scenario blocks to ensure comparable preconditions
    gw_texts = [(b.get("gw") or "").strip() for b in blocks]
    gw_embeddings = model.encode(gw_texts, convert_to_tensor=True, normalize_embeddings=True) if gw_texts else None
    gw_sim_matrix = util.cos_sim(gw_embeddings, gw_embeddings) if gw_embeddings is not None else None
    gw_sim_thresh = float(os.environ.get("INTRA_GW_SIM_FOR_NLI", "0.90"))

    two_stage = str(os.environ.get("THEN_TWO_STAGE", "1")).lower() in ("1", "true", "yes", "on")
    for idx_a in range(len(raw_then)):
        for idx_b in range(idx_a + 1, len(raw_then)):
            similarity_score = float(similarity_matrix[idx_a, idx_b].item())
            if similarity_score < similarity_threshold:
                continue
            # Preconditions must be similar enough
            if gw_sim_matrix is not None:
                ba = origin_block_idx[idx_a]
                bb = origin_block_idx[idx_b]
                if float(gw_sim_matrix[ba, bb].item()) < gw_sim_thresh:
                    continue
            token_overlap = len(_token_set(raw_then[idx_a]) & _token_set(raw_then[idx_b]))
            if token_overlap < min_token_overlap:
                continue

            max_contradiction_prob = 0.0
            max_entailment_prob = 0.0
            try:
                premise = enriched_then[idx_a]
                hypothesis = enriched_then[idx_b]
                entailment_prob = nli_entailment_prob(premise, hypothesis)
                contradiction_prob = nli_contradiction_prob(premise, hypothesis)
                max_entailment_prob = max(max_entailment_prob, entailment_prob)
                max_contradiction_prob = max(max_contradiction_prob, contradiction_prob)
            except Exception:
                pass

            if (max_contradiction_prob >= contradiction_threshold) and \
               ((max_contradiction_prob - max_entailment_prob) >= nli_margin):
                # Optional second-stage filter: validate contradiction on raw Then only
                if two_stage:
                    try:
                        raw_premise = raw_then[idx_a]
                        raw_hyp = raw_then[idx_b]
                        raw_e = nli_entailment_prob(raw_premise, raw_hyp)
                        raw_c = nli_contradiction_prob(raw_premise, raw_hyp)
                        if (raw_c >= float(os.environ.get("NLI2_CONTRA_THRESHOLD", "0.68"))) and \
                           ((raw_c - raw_e) >= float(os.environ.get("NLI2_MARGIN", "0.18"))):
                            contradiction_weight_sum += 1.5 if max_contradiction_prob >= 0.80 else 1.0
                        else:
                            # filtered out by raw-Then gate
                            pass
                    except Exception:
                        # if second-stage fails, be conservative and keep original
                        contradiction_weight_sum += 1.5 if max_contradiction_prob >= 0.80 else 1.0
                else:
                    contradiction_weight_sum += 1.5 if max_contradiction_prob >= 0.80 else 1.0

    return contradiction_weight_sum


def _extract_gw_then_blocks_with_context(text: str) -> List[Dict[str, Any]]:
    """Extract scenario blocks with accumulated Given/When (GW) text and Then lines.
    GW collects all non-tag/comment lines outside Then; Then collects "Тогда/Then" and subsequent "И/And".
    """
    blocks: List[Dict[str, Any]] = []
    current_gw: List[str] = []
    current_then: List[str] = []
    in_then = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # New scenario boundary
        if re.match(r"^(Сценарий:|Scenario:)\s*(.+)$", line, re.IGNORECASE):
            if current_gw or current_then:
                blocks.append({"gw": "\n".join(current_gw), "then": current_then[:]})
            current_gw = []
            current_then = []
            in_then = False
            continue
        # Then block
        m_then = re.match(r"^(Тогда|Then)\s+(.+)$", line, re.IGNORECASE)
        if m_then:
            current_then.append(m_then.group(2).strip())
            in_then = True
            continue
        if in_then:
            m_and = re.match(r"^(И|And)\s+(.+)$", line, re.IGNORECASE)
            if m_and:
                current_then.append(m_and.group(2).strip())
                continue
            else:
                in_then = False
        # Accumulate GW text when not in Then; skip tags/comments
        if not re.match(r"^(@|#)", line):
            current_gw.append(line)
    if current_gw or current_then:
        blocks.append({"gw": "\n".join(current_gw), "then": current_then[:]})
    return blocks


