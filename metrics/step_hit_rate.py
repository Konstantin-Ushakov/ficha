import re
from typing import Dict, List, Tuple


def _extract_then_blocks(text: str) -> List[str]:
    """Return normalized Then/And lines across feature text.

    This is a lightweight structural view, without GW context.
    """
    out: List[str] = []
    in_then = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^(Сценарий:|Scenario:|Предыстория:|Background:)", line, re.IGNORECASE):
            in_then = False
            continue
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


def compute_step_hit_rate(
    req_units: List[str],
    feature_text: str,
    min_keyword_overlap: int = 2,
) -> Tuple[float, Dict[str, List]]:
    """Step-level coverage using keyword overlap against Then/And lines.

    What it measures (plain words):
      Насколько требования .md "звучат" так же, как и формулировки проверок в шагах Then/И.
      Если у требования и у какого-то Then/И есть хотя бы `min_keyword_overlap` общих слов (по токенам),
      считаем, что это требование покрыто на уровне шага.

    Intended use:
      - Простая и объяснимая прокси-метрика структурного покрытия.
      - Помогает увидеть требования, которые не отражены в формулировках Then/И теми же терминами.

    Returns:
      (step_hit_rate, details) where details contains:
        - step_hits_idx: indices of requirements covered by any Then/And line
        - step_misses_idx: indices without such coverage
        - intersections: list[(i, overlap_count)] — максимальное пересечение по токенам для требования i
        - step_best_then: list[(i, then_text, overlap_count)] — лучший Then по пересечению
        - step_token_overlap: list[(i, shared_tokens:list, req_only:list, then_only:list)]
    """
    details: Dict[str, List] = {"step_hits_idx": [], "step_misses_idx": [], "intersections": [], "step_best_then": [], "step_token_overlap": []}
    if not req_units:
        return 0.0, details
    thens = _extract_then_blocks(feature_text)
    if not thens:
        return 0.0, details

    def toks(s: str) -> set[str]:
        return set(re.findall(r"[\w\-/]+", s.lower()))

    hits = 0
    for i, r in enumerate(req_units):
        tr = toks(r)
        best_inter = 0
        best_then = ""
        best_shared: List[str] = []
        best_then_only: List[str] = []
        best_req_only: List[str] = []
        for t in thens:
            tt = toks(t)
            shared = sorted(list(tr & tt))
            inter = len(shared)
            if inter > best_inter:
                best_inter = inter
                best_then = t
                best_shared = shared
                best_then_only = sorted(list(tt - tr))
                best_req_only = sorted(list(tr - tt))
            if inter >= min_keyword_overlap:
                hits += 1
                details["step_hits_idx"].append(i)
                # continue scanning to find best overlap for reporting
        if best_inter < min_keyword_overlap:
            details["step_misses_idx"].append(i)
        details["intersections"].append((i, best_inter))
        details["step_best_then"].append((i, best_then, best_inter))
        details["step_token_overlap"].append((i, best_shared, best_req_only, best_then_only))

    step_hit_rate = hits / max(1, len(req_units))
    return step_hit_rate, details


def get_description() -> str:
    return (
        "step_hit_rate — доля требований .md, для которых нашлась строка Then/И с ≥2 общими токенами. "
        "Показывает структурное покрытие проверок на уровне формулировок шагов."
    )


def get_detailed_fix() -> str:
    return (
        "Сделайте Then/И конкретными и терминологически согласованными с .md (те же слова для сущностей и результатов). "
        "Избегайте общих формулировок без ключевых слов из требований."
    )

