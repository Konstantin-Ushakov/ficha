from typing import Tuple
import os
from .model import get_model
from .scoring import compute_document_alignment_score
from .contradictions import compute_contradictions
from .coverage import compute_coverage, compute_goal_alignment
from .extras import compute_extras
from .multi_intent import compute_multi_intent_metrics


def evaluate_pass(md_text: str, feat_text: str) -> Tuple[bool, dict]:
    """End-to-end evaluation from texts only. Returns (passed, details_dict).

    Decision policy (high level):
      1) Block on contradictions (NLI/structural), with soft promotion from multi-intent + signals.
      2) Otherwise require: score ≥ THRESHOLD and (hit_rate ≥ min_hit_rate_dynamic) and
         semantic_coverage via kw/nli/step (with relaxations when agg_base is high), OR pass backstop paths.

    semantic_coverage definition:
      semantic_ok := (kw_hit_rate ≥ 0.60) OR (nli_hit_rate ≥ 0.60) OR (step_hit_rate ≥ 0.60)
      If agg_base ≥ 0.82, allow weaker keyword path: kw_hit_rate ≥ 0.40.

    Notes on placeholders (current implementation):
      nli_hit_rate and step_hit_rate are computed as 0.0 in coverage; kw_hit_rate is the active path.
      This design allows extending coverage without changing evaluation logic.
    """
    model = get_model()
    # Toggle individual signals via ENV (exported from config.metrics)
    enable_goal = str(os.environ.get("GOAL_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    enable_contra = str(os.environ.get("CONTRA_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    enable_multi_intent = str(os.environ.get("MULTI_INTENT_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
    agg_base, cos, score = compute_document_alignment_score(md_text, feat_text, model)

    # Coverage via embeddings (moved to coverage module)
    min_hit_rate = float(os.environ.get("MIN_HIT_RATE", "0.67"))
    cov = compute_coverage(md_text, feat_text, model)
    goal = compute_goal_alignment(md_text, feat_text, model) if enable_goal else {"goal_sim": 0.0, "goal_hit": False, "goal_worst": []}
    # Extra metrics (opt-in via ENV flags)
    extras = compute_extras(md_text, feat_text, cov)
    hit_rate = cov.get("hit_rate", 0.0)
    keyword_hit_rate = cov.get("kw_hit_rate", 0.0)
    entailment_hit_rate = cov.get("nli_hit_rate", 0.0)
    step_hit_rate = cov.get("step_hit_rate", 0.0)
    goal_sim = goal.get("goal_sim", 0.0)
    goal_hit = goal.get("goal_hit", False)
    goal_worst = goal.get("goal_worst", [])

    # Contradictions from structural + NLI Then
    contra_flag, contra_details = compute_contradictions(md_text, feat_text, model) if enable_contra else (False, {})

    # Dynamic threshold
    min_hit_rate_dynamic = min_hit_rate
    if agg_base >= 0.70:
        min_hit_rate_dynamic = min(min_hit_rate_dynamic, 0.58)
    # Further relax hit_rate when doc alignment is high but keyword coverage is low
    if agg_base >= 0.85 and keyword_hit_rate <= 0.25:
        min_hit_rate_dynamic = min(min_hit_rate_dynamic, float(os.environ.get("MIN_HIT_RATE_RELAX1", "0.56")))
    if agg_base >= 0.90 and keyword_hit_rate <= 0.20:
        min_hit_rate_dynamic = min(min_hit_rate_dynamic, float(os.environ.get("MIN_HIT_RATE_RELAX2", "0.52")))

    # Multi-intent on MD
    multi_intent = compute_multi_intent_metrics(md_text, model) if enable_multi_intent else {"multi_intent": False}
    # Soft safety-net: multi-intent + any structural signal => contradiction
    if not contra_flag and bool(multi_intent.get("multi_intent")):
        signals = 0
        signals += 1 if contra_details.get("then_neg", 0.0) >= float(os.environ.get("MI_THEN_NEG_MIN", "1.0")) else 0
        signals += 1 if contra_details.get("gw_div", 0.0) >= float(os.environ.get("MI_GW_DIV_MIN", "1.0")) else 0
        signals += 1 if contra_details.get("title_div", 0.0) >= float(os.environ.get("MI_TITLE_DIV_MIN", "1.0")) else 0
        signals += 1 if contra_details.get("adj_div", 0.0) >= float(os.environ.get("MI_ADJ_DIV_MIN", "1.0")) else 0
        if signals >= int(os.environ.get("MI_SIGNALS_MIN", "1")):
            contra_flag = True

    # Decision (универсальный: любая включенная метрика провалилась → вся фича не пройдена)
    # Готовим список метрик со статусами
    metrics = []
    # Вспомогательные функции
    def _flag(env_key: str, default: str = "1") -> bool:
        return str(os.environ.get(env_key, default)).lower() in ("1", "true", "yes", "on")

    def _policy_for(name: str, default_policy: str = "min") -> str:
        return str(os.environ.get(f"METRIC_{name.upper()}_POLICY", default_policy)).lower()

    def _thresh_for(name: str, default_value: float | None) -> float | None:
        # Ищем универсальные переменные окружения
        val = os.environ.get(f"METRIC_{name.upper()}_THRESH")
        if val is None:
            val = os.environ.get(f"THRESH_{name.upper()}")
        try:
            return float(val) if val is not None else default_value
        except Exception:
            return default_value

    # Всегда проверяем score
    score_thr = float(os.environ.get("THRESHOLD", "0.75")) * 100.0
    metrics.append({
        "name": "score",
        "value": score,
        "policy": "min",
        "threshold": score_thr,
    })

    # hit_rate (если включен)
    if _flag("HIT_ENABLE", "1"):
        metrics.append({
            "name": "hit_rate",
            "value": hit_rate,
            "policy": "min",
            "threshold": min_hit_rate_dynamic,
        })

    # kw_hit_rate (если включен)
    if _flag("KW_HIT_ENABLE", "1"):
        kw_min = _thresh_for("kw_hit_rate", 0.60)
        # Смягчение порога при высоком agg_base
        if agg_base >= 0.82:
            kw_min = min(kw_min, float(os.environ.get("KW_HIT_RELAX", "0.40")))
        metrics.append({
            "name": "kw_hit_rate",
            "value": keyword_hit_rate,
            "policy": "min",
            "threshold": kw_min,
        })

    # nli_hit_rate (если включен)
    if _flag("NLI_HIT_ENABLE", "1"):
        metrics.append({
            "name": "nli_hit_rate",
            "value": entailment_hit_rate,
            "policy": "min",
            "threshold": _thresh_for("nli_hit_rate", 0.60),
        })

    # step_hit_rate (если включен)
    if _flag("STEP_HIT_ENABLE", "1"):
        metrics.append({
            "name": "step_hit_rate",
            "value": step_hit_rate,
            "policy": "min",
            "threshold": _thresh_for("step_hit_rate", 0.60),
        })

    # goal_sim (если включен)
    if enable_goal:
        metrics.append({
            "name": "goal_sim",
            "value": goal_sim,
            "policy": "min",
            "threshold": float(os.environ.get("GOAL_SIM_THRESHOLD", "0.60")),
        })

    # contradictions (если включен): должно быть 0.0
    if enable_contra:
        metrics.append({
            "name": "contradictions",
            "value": 1.0 if contra_flag else 0.0,
            "policy": "max",
            "threshold": 0.0,
        })

    # Прочие метрики из extras: включаем в проверку только если задан универсальный порог через ENV
    for k, v in extras.items():
        thr = _thresh_for(k, None)
        if thr is None:
            continue  # нет порога → не блокируем
        pol = _policy_for(k, "min")
        metrics.append({
            "name": k,
            "value": float(v),
            "policy": pol,
            "threshold": thr,
        })

    # Вычисляем статусы по метрикам
    all_status = True
    for m in metrics:
        policy = m.get("policy", "min")
        val = float(m.get("value", 0.0))
        thr = float(m.get("threshold", 0.0))
        if policy == "max":
            ok = (val <= thr + 1e-9)
        else:
            ok = (val + 1e-9) >= thr
        m["status"] = bool(ok)
        all_status = all_status and bool(ok)

    passed = all_status
    semantic_branch = "blocked_contradiction" if contra_flag else ("" if passed else "none")

    details = {
        "agg_base": agg_base,
        "cos": cos,
        "score": score,
        "hit_rate": hit_rate,
        "kw_hit_rate": keyword_hit_rate,
        "nli_hit_rate": entailment_hit_rate,
        "step_hit_rate": step_hit_rate,
        "contra_flag": contra_flag,
            "min_hit_rate_dynamic": min_hit_rate_dynamic,
        "multi_intent": multi_intent,
            "semantic_branch": semantic_branch,
        **{f"contra_{k}": v for k, v in contra_details.items()},
            "goal_sim": goal_sim,
            "goal_hit": goal_hit,
            "goal_worst": goal_worst,
            # Упакованные статусы метрик (универсальный формат)
            "metrics": metrics,
            "all_metrics_passed": passed,
            # Coverage debug/explain fields (used by CLI output)
            "req_units": cov.get("req_units", []),
            "best_match_indices": cov.get("best_match_indices", []),
            "best_match_sims": cov.get("best_match_sims", []),
            "best_match_texts": cov.get("best_match_texts", []),
            "misses_idx": cov.get("misses_idx", []),
            "kw_misses_idx": cov.get("kw_misses_idx", []),
            "nli_not_entailed_idx": cov.get("nli_not_entailed_idx", []),
            "nli_best_pairs": cov.get("nli_best_pairs", []),
            "nli_best_top3_pairs": cov.get("nli_best_top3_pairs", []),
            "nli_worst_pairs": cov.get("nli_worst_pairs", []),
            "nli_entail_threshold": cov.get("nli_entail_threshold", float(os.environ.get("NLI_HIT_ENTAIL", "0.70"))),
            "step_misses_idx": cov.get("step_misses_idx", []),
            "step_best_then": cov.get("step_best_then", []),
            "step_token_overlap": cov.get("step_token_overlap", []),
            # Extras flattened in details for reporting
            **extras,
    }
    return passed, details


