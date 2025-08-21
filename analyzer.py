import os
import glob
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json


def load_text(p: Path) -> str:
    return p.read_text(encoding='utf-8')


def concat_features(feature_files: List[Path]) -> str:
    return "\n\n".join(load_text(p) for p in feature_files)


def parse_category_mode() -> tuple[str | None, int | None, str]:
    """Parse TEST_CATEGORY env and return (mode, number, raw).
    Supported:
      - 'till-X' → ("till", X, raw)
      - everything else → (None, None, raw)
    Numeric categories (e.g., "10") are NOT handled here on purpose; use tags.
    """
    raw = os.environ.get("TEST_CATEGORY", "").strip()
    if not raw:
        return None, None, raw
    # numeric categories intentionally ignored to rely on tags
    if raw.startswith("till-"):
        tail = raw.split("-", 1)[1]
        if tail.isdigit():
            try:
                return "till", int(tail), raw
            except Exception:
                return None, None, raw
    return None, None, raw


def _strict_should_fail_enabled() -> bool:
    return str(os.environ.get("STRICT_SHOULD_FAIL", "0")).lower() in ("1", "true", "yes", "on")


def compute_effective_passed(expected_fail: bool, raw_passed: bool) -> bool:
    """Return effective passed value with optional inversion for should-fail cases.

    - If STRICT_SHOULD_FAIL is on and this is an expected-fail case, invert raw result.
    - Otherwise return raw result unchanged.
    """
    if _strict_should_fail_enabled() and expected_fail:
        return (not raw_passed)
    return raw_passed


def is_unexpected_outcome(expected_fail: bool, raw_passed: bool) -> bool:
    """Determine whether outcome is unexpected.

    Compare actual (effective) outcome with desired outcome:
    - For positive specs: desired pass = True
    - For should-fail specs:
        - strict mode ON → desired pass = True (инверсия: «упал» считается «пройдено»)
        - strict mode OFF → desired pass = False (обычное ожидание провала)
    """
    actual_passed = compute_effective_passed(expected_fail=expected_fail, raw_passed=raw_passed)
    desired_passed = (not expected_fail) or (_strict_should_fail_enabled() and expected_fail)
    return actual_passed != desired_passed


def _collect_fail_reasons(res: Dict[str, Any], threshold: float, threshold_pct: int) -> List[str]:
    """Build human-readable reasons for failure based on metrics in res."""
    reasons: List[str] = []
    # If this is an expected-fail case and STRICT_SHOULD_FAIL is ON, we do not collect reasons
    # because such specs are not considered failures in strict mode (they are intended negatives).
    try:
        strict_on = str(os.environ.get("STRICT_SHOULD_FAIL", "0")).lower() in ("1", "true", "yes", "on")
    except Exception:
        strict_on = False
    if strict_on and bool(res.get("expected_fail", False)):
        return reasons
    hit_rate = res.get('hit_rate', 0.0)
    score = res.get('score', 0.0)
    agg_base = res.get('agg_base', 0.0)
    min_hit_rate_logged = res.get('min_hit_rate_dynamic', float(os.environ.get("MIN_HIT_RATE", "0.67")))
    base_ok = (hit_rate + 1e-6) >= min_hit_rate_logged
    kw = res.get('kw_hit_rate', 0.0)
    nli = res.get('nli_hit_rate', 0.0)
    step = res.get('step_hit_rate', 0.0)
    semantic_ok = (kw >= 0.60) or (nli >= 0.60) or (step >= 0.60)
    if agg_base >= 0.82:
        semantic_ok = semantic_ok or (kw >= 0.40)
    if score < (threshold * 100.0):
        reasons.append(f"score {score:.0f}% < {threshold_pct}%")
    if (hit_rate + 1e-6) < min_hit_rate_logged:
        reasons.append(f"hit_rate {hit_rate:.2f} < {min_hit_rate_logged:.2f}")
    if base_ok and (not semantic_ok):
        reasons.append(
            f"semantic_coverage слабое: kw={kw:.2f}, nli={nli:.2f}, step={step:.2f} (требуются ≥0.60; при agg_base≥0.82 — kw≥0.40)"
        )
    # Structural contradictions detail (if any)
    if res.get('contra_flag', False):
        srcs: List[str] = []
        if (res.get('contra_doc_then', 0.0) or 0.0) > 0.0:
            srcs.append("Then-NLI")
        if (res.get('contra_then_neg', 0.0) or 0.0) > 0.0:
            srcs.append("Then-NEG")
        if (res.get('contra_gw_div', 0.0) or 0.0) > 0.0:
            srcs.append("GW-div")
        if (res.get('contra_title_div', 0.0) or 0.0) > 0.0:
            srcs.append("Title-div")
        if (res.get('contra_adj_div', 0.0) or 0.0) > 0.0:
            srcs.append("Adj-div")
        srcs_str = ", ".join(srcs) if srcs else "unknown"
        reasons.append(f"contradiction (sources: {srcs_str})")
    if not reasons:
        reasons.append("не пройдено без явной причины (см. подробный отчёт выше)")
    return reasons


def main():
    # Detect repository root (works in Docker and local runs)
    base = Path("/workspace")
    try:
        candidates = [
            base,
            Path.cwd(),
            Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else None,
            Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) >= 2 else None,
            Path(__file__).resolve().parent.parent,  # tests
            Path(__file__).resolve().parent.parent.parent,  # repo root (fallback)
        ]
        for cand in candidates:
            if cand is None:
                continue
            try:
                if (cand / "tests" / "analyze").exists() and (cand / "tests" / "bdd-orchestrator").exists():
                    base = cand
                    break
            except Exception:
                continue
    except Exception:
        pass
    # Load analyzer config and export to env BEFORE importing metrics to ensure precedence over .env
    cfg_used = None
    # Load analyzer config and export to env BEFORE importing metrics to ensure precedence over .env
    cfg_used = None
    cfg_paths = [
        Path("/app/config.json"),  # Docker container path
        base / "tests" / "analyze" / "config.json",  # Mounted workspace path
        base / "tests" / "analyze" / "config.example.jsonc",  # Fallback example
    ]
    def _export_cfg(obj: dict):
        # Export flat primitives
        for k, v in obj.items():
            if k == "metrics" and isinstance(v, dict):
                # Flatten nested metrics categories → ENV flags XXX=1/0
                for _cat_name, toggles in v.items():
                    if not isinstance(toggles, dict):
                        continue
                    for key, val in toggles.items():
                        if isinstance(val, bool):
                            os.environ[str(key)] = "1" if val else "0"
                        elif isinstance(val, (str, int, float)):
                            os.environ[str(key)] = str(val)
                continue
            if isinstance(v, (str, int, float)):
                os.environ[str(k)] = str(v)

    for cfg in cfg_paths:
        try:
            if cfg.exists():
                # Strip JSONC comments (// and /* */) before json.loads
                raw_text = cfg.read_text(encoding="utf-8")
                try:
                    import re as _re_jsonc
                    # remove // comments
                    raw_text = _re_jsonc.sub(r"//.*?$", "", raw_text, flags=_re_jsonc.MULTILINE)
                    # remove /* ... */ comments
                    raw_text = _re_jsonc.sub(r"/\*.*?\*/", "", raw_text, flags=_re_jsonc.DOTALL)
                except Exception:
                    pass
                data = json.loads(raw_text)
                _export_cfg(data)
                cfg_used = str(cfg)
                break
        except Exception:
            pass
    # TRACE: unique run id
    run_id = str(uuid.uuid4())
    try:
        print(f"[TRACE] analyzer_run_id={run_id} started_at={datetime.utcnow().isoformat()}Z TEST_CATEGORY={os.environ.get('TEST_CATEGORY','')} STRICT_SHOULD_FAIL={os.environ.get('STRICT_SHOULD_FAIL','')} MODE={'ANALYZE'}")
    except Exception:
        pass

    if cfg_used:
        try:
            print(f"[ANALYZER] Config loaded: {cfg_used} (values override .env for this run)")
            # Print enabled metrics overview (by categories)
            def _is_on(k: str) -> bool:
                return str(os.environ.get(k, "0")).lower() in ("1","true","yes","on")
            categories = {
                "feature_definition": [
                    ("TERMINOLOGY_CONSISTENCY_ENABLE", "terminology_consistency"),
                    ("FVI_MIT_ENABLE", "fvi_mit"),
                    ("MULTI_INTENT_ENABLE", "multi_intent"),
                ],
                "md_duplicates": [
                    ("DUPS_BM25_ENABLE", "md_duplicates_bm25"),
                    ("DUPS_TFIDF_ENABLE", "md_duplicates_tfidf"),
                ],
                "coverage_all": [
                    ("HIT_ENABLE", "hit_rate"),
                    ("KW_HIT_ENABLE", "kw_hit_rate"),
                    ("NLI_HIT_ENABLE", "nli_hit_rate"),
                    ("STEP_HIT_ENABLE", "step_hit_rate"),
                    ("JACCARD_LEMMA_ENABLE", "jaccard_lemma_overlap"),
                    ("CARDINALITY_ALIGNMENT_ENABLE", "cardinality_alignment"),
                    ("BM25_ENABLE", "bm25okapi"),
                    ("TFIDF_ENABLE", "tfidf_cosine"),
                    ("GRAPH_COVERAGE_ENABLE", "graph_coverage"),
                ],
                "per_scenario": [
                    ("PER_SCENARIO_ENABLE", "per_scenario_alignment"),
                    ("TRACE_DENSITY_ENABLE", "trace_density"),
                ],
                "quality_structure": [
                    ("FEATURE_HEADER_VALID_ENABLE", "feature_header_valid"),
                    ("DUP_SCENARIOS_ENABLE", "duplicate_scenarios_ratio"),
                    ("GOAL_ENABLE", "goal_alignment"),
                    ("CONTRA_ENABLE", "contradictions"),
                ],
            }
            print("[METRICS] enabled:")
            for cat, pairs in categories.items():
                enabled = [name for (flag,name) in pairs if _is_on(flag)]
                print(f"  - {cat}: {', '.join(enabled) if enabled else '<none>'}")
        except Exception:
            pass

    # Deferred imports that depend on config
    from metrics.evaluation import evaluate_pass
    from metrics.md_duplicates import compute_md_duplicates
    from core.models import MODEL_NAME
    # threshold from env; default 0.75 (75%)
    try:
        threshold = float(os.environ.get("THRESHOLD", "0.75"))
    except ValueError:
        threshold = 0.75
    threshold_pct = int(round(threshold * 100))
    # Prefer mounted paths when running in Docker demo
    mounted_user_features = base / "user-features"
    if mounted_user_features.exists():
        md_glob = str(mounted_user_features / "*_*.md")
    else:
        md_glob = str(base / "user-features" / "*_*.md")
    md_paths = sorted(Path(p) for p in glob.glob(md_glob))

    mode, cat_num, _raw = parse_category_mode()
    if mode in ("till",) and cat_num is not None:
        filtered = []
        for p in md_paths:
            stem = p.stem
            try:
                pref = stem.split("_", 1)[0]
                if not pref.isdigit():
                    continue
                pref_i = int(pref)
                if (mode == "till" and pref_i <= cat_num):
                    filtered.append(p)
            except Exception:
                continue
        md_paths = filtered
    # Tag-based filtering: if TEST_CATEGORY is provided and not a till-X, treat it as a tag (e.g., '10', 'backend-api', 'auth-simp')
    raw_tag = os.environ.get("TEST_CATEGORY", "").strip()
    use_tag_filter = (raw_tag != "") and (mode is None)
    if use_tag_filter:
        try:
            print(f"[ANALYZER] Using tag filter: @{raw_tag}")
        except Exception:
            pass

    failed: List[str] = []
    results: List[Dict[str, Any]] = []
    scenarios_total: int = 0

    # Precompute MD duplicates if enabled
    try:
        if str(os.environ.get("BM25_ENABLE", "0")).lower() in ("1","true","yes","on") or str(os.environ.get("TFIDF_ENABLE", "0")).lower() in ("1","true","yes","on"):
            md_pairs = []
            for _p in md_paths:
                try:
                    md_text = load_text(_p)
                except Exception:
                    md_text = ""
                md_pairs.append((_p.stem, md_text))
            dup_map = compute_md_duplicates(md_pairs)
        else:
            dup_map = {}
    except Exception:
        dup_map = {}

    for md in md_paths:
        name = md.stem
        try:
            _cat, slug = name.split("_", 1)
        except ValueError:
            slug = name
        cat_int = None
        if _cat.isdigit():
            try:
                cat_int = int(_cat)
            except Exception:
                cat_int = None
        feat_files = []
        # Discover feature files
        features_dir_candidates = [
            base / "tests" / "bdd-orchestrator" / "features",
            base / "tests",  # demo mount: features directly in /workspace/tests
        ]
        # Prefer explicit mapping from MD: section "## Связь с файлами" -> "### Реализация"
        md_text = load_text(md)
        try:
            lines = md_text.splitlines()
            # Find start of "## Связь с файлами"
            start_sf = None
            for i, raw in enumerate(lines):
                if raw.strip().lower().startswith("## связь с файлами"):
                    start_sf = i
                    break
            start_impl = None
            if start_sf is not None:
                for j in range(start_sf + 1, len(lines)):
                    t = lines[j].strip().lower()
                    if t.startswith("### реализация"):
                        start_impl = j + 1
                        break
                    if t.startswith("## "):
                        break
            if start_impl is not None:
                # Collect bullet lines until next heading (## or ###) or a blank separation
                for k in range(start_impl, len(lines)):
                    raw = lines[k]
                    strip = raw.strip()
                    if not strip:
                        continue
                    low = strip.lower()
                    if low.startswith("## ") or low.startswith("### "):
                        break
                    # match any bullet and capture first path ending with .feature
                    import re as _re
                    m = _re.search(r"[\-*•]\s+(.+?\.feature)\b", strip)
                    if m:
                        rel = m.group(1).strip()
                        # normalize path
                        try:
                            p = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
                        except Exception:
                            p = Path(rel)
                        if p.exists() and p.suffix == ".feature":
                            feat_files.append(p)
            # Fallback to slug-based discovery if nothing found
        except Exception:
            feat_files = []
        if not feat_files:
            for features_dir in features_dir_candidates:
                for cat_prefix in [str(i) for i in range(0, 11)]:
                    if mode == "till" and cat_num is not None:
                        try:
                            if int(cat_prefix) > cat_num:
                                continue
                        except Exception:
                            pass
                    candidate = features_dir / f"{cat_prefix}_{slug}.feature"
                    if candidate.exists():
                        feat_files.append(candidate)
                if feat_files:
                    break

        # Apply tag filter to feature files if requested
        if use_tag_filter and feat_files:
            tag_pattern = re.compile(rf"(^|\s)@{re.escape(raw_tag)}(\s|$)")
            filtered_feat_files = []
            for fp in feat_files:
                try:
                    ft = load_text(Path(fp))
                    if tag_pattern.search(ft):
                        filtered_feat_files.append(fp)
                except Exception:
                    continue
            feat_files = filtered_feat_files

        if not feat_files:
            # If filtering by tag, silently skip specs without this tag
            if use_tag_filter:
                continue
            # Otherwise, record as missing features with empty result stub
            results.append({
                "name": name,
                "score": 0.0,
                "files": [],
                "cos": 0.0,
                "hit_rate": 0.0,
                "kw_hit_rate": 0.0,
                "nli_hit_rate": 0.0,
                "step_hit_rate": 0.0,
                "contra_flag": False,
                "feature_contradiction": 0.0,
                "contra_doc_then": 0.0,
                "contra_then_neg": 0.0,
                "contra_gw_div": 0.0,
                "contra_title_div": 0.0,
                "contra_adj_div": 0.0,
                "md_nli_contra": 0.0,
                "expected_fail": False,
                "raw_passed": False,
                "agg_base": 0.0,
                "min_hit_rate_dynamic": float(os.environ.get("MIN_HIT_RATE", "0.67")),
                "multi_intent": {},
            })
            if is_unexpected_outcome(expected_fail=False, raw_passed=False):
                failed.append(name)
            continue

        # md_text already loaded above
        feat_text = concat_features(feat_files)
        # Подсчёт сценариев (RU/EN) по всем привязным feature-файлам
        try:
            scenarios_total += len(re.findall(r"^\s*(Сценарий|Scenario):", feat_text, flags=re.MULTILINE))
        except Exception:
            pass
        # Detect expected-fail ONLY by tag in feature files (ignore filename)
        lower_name = name.lower()
        is_example = ("_example-" in lower_name) or lower_name.startswith("example-") or lower_name.startswith("0_example-")
        expected_fail = False
        try:
            tag_should_fail = re.compile(r"(^|\s)@should-fail(\s|$)")
            tag_example = re.compile(r"(^|\s)@(?:0-)?example(\s|$)")
            for fp in feat_files:
                ft = load_text(Path(fp))
                if not expected_fail and tag_should_fail.search(ft):
                    expected_fail = True
                if (not is_example) and tag_example.search(ft):
                    is_example = True
        except Exception:
            pass

        # Evaluate entirely inside metrics.evaluation
        passed, det = evaluate_pass(md_text=md_text, feat_text=feat_text)
        score = det.get("score", 0.0)
        agg_base = det.get("agg_base", 0.0)
        hit_rate = det.get("hit_rate", 0.0)
        keyword_hit_rate = det.get("kw_hit_rate", 0.0)
        entailment_hit_rate = det.get("nli_hit_rate", 0.0)
        step_hit_rate = det.get("step_hit_rate", 0.0)
        goal_sim = det.get("goal_sim", 0.0)
        goal_hit = det.get("goal_hit", False)
        goal_worst = det.get("goal_worst", [])
        contra_flag = det.get("contra_flag", False)
        cos = det.get("cos", min(1.0, score / 100.0))
        # Keep raw outcome from evaluator; inversion applies only when registering outcome
        raw_passed = passed

        res_item = {
            "name": name,
            "score": score,
            "files": [str(p) for p in feat_files],
            "doc": str(md),
            "cos": cos,
            "hit_rate": hit_rate,
            "kw_hit_rate": keyword_hit_rate,
            "nli_hit_rate": entailment_hit_rate,
            "step_hit_rate": step_hit_rate,
            "goal_sim": goal_sim,
            "goal_hit": goal_hit,
            "contra_flag": contra_flag,
            "feature_contradiction": det.get("contra_feature_contradiction", 0.0),
            "contra_doc_then": det.get("contra_doc_then_contra", 0.0),
            "contra_then_neg": det.get("contra_then_neg", 0.0),
            "contra_gw_div": det.get("contra_gw_div", 0.0),
            "contra_title_div": det.get("contra_title_div", 0.0),
            "contra_adj_div": det.get("contra_adj_div", 0.0),
            "md_nli_contra": det.get("contra_md_nli_contra", 0.0),
            "expected_fail": expected_fail,
            "raw_passed": raw_passed,
            "agg_base": agg_base,
            "min_hit_rate_dynamic": det.get("min_hit_rate_dynamic", float(os.environ.get("MIN_HIT_RATE", "0.67"))),
            "multi_intent": det.get("multi_intent", {}),
            "semantic_branch": det.get("semantic_branch", ""),
            "details": det,
        }
        # attach duplicates info for this md if any
        if dup_map and name in dup_map:
            res_item["md_duplicates"] = dup_map.get(name, [])
        results.append(res_item)

        # Single point of truth for unexpected outcomes with should-fail inversion
        if is_unexpected_outcome(expected_fail=expected_fail, raw_passed=raw_passed):
            failed.append(name)

    print("=== РЕЗУЛЬТАТЫ СЕМАНТИЧЕСКОГО ПОКРЫТИЯ ===")
    print(f"Модель: {MODEL_NAME}")
    strict_should_fail = _strict_should_fail_enabled()

    # helper: find first locations of snippet in feature files
    def _find_locations(snippet: str, files: List[str]) -> List[str]:
        out: List[str] = []
        if not snippet or not files:
            return out
        s_low = snippet.strip().lower()
        for fp in files:
            try:
                p = Path(fp)
                lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
                for idx, line in enumerate(lines, start=1):
                    if s_low and s_low in line.lower():
                        out.append(f"{fp}:{idx}")
                        break
            except Exception:
                continue
        return out
    for res in results:
        det = res.get('details', {}) or {}
        print(f"{res['name']}: {res['score']:.1f}%")
        if res["files"]:
            print("  features:")
            for f in res["files"]:
                print(f"    - {f}")
        print(f"  model_cosine: {res['cos']*100.0:.1f}%")
        try:
            print(f"  agg_base: {res.get('agg_base', 0.0)*100.0:.1f}%")
        except Exception:
            pass
        try:
            print(f"  min_hit_rate (эффективный): {res.get('min_hit_rate_dynamic', 0.0):.2f}")
        except Exception:
            pass
        mi = res.get('multi_intent') or {}
        mi_str = f"  multi_intent: {mi.get('multi_intent', False)} (clusters={mi.get('topic_clusters', 0)}, main={mi.get('main_cluster_ratio', 0.0):.2f}, non_behavioral={mi.get('non_behavioral_ratio', 0.0):.2f})"
        md_enforce = str(os.environ.get("MD_CONTRA_ENFORCE", "1")).lower() in ("1", "true", "yes", "on")
        # Desired outcome depends on strict should-fail mode (use per-result flag)
        expected_fail_res = bool(res.get('expected_fail', False))
        desired_passed = (not expected_fail_res) or (strict_should_fail and expected_fail_res)
        expect_str = "Ожидаем: успех" if desired_passed else "Ожидаем: неудачу"
        # Прошел тест или нет
        result_evaluation = compute_effective_passed(expected_fail=res['expected_fail'], raw_passed=bool(res.get('raw_passed', False)))
        passed_str = "Итог: провал" if (not result_evaluation) else "Итог: пройдено"
        # Добавляем краткую пометку ТОЛЬКО для действительно неожиданных исходов
        unexpected = is_unexpected_outcome(expected_fail=expected_fail_res, raw_passed=bool(res.get('raw_passed', False)))
        if unexpected:
            if strict_should_fail and expected_fail_res:
                passed_str += " — Ожидается неудача (strict @should-fail), но тест прошел"
            else:
                passed_str += " — Ожидается прохождение, но тест не прошел"
        contra_text = "Противоречия: есть" if res['contra_flag'] else "Противоречия: нет"
        # Уточнение источников противоречий
        if res['contra_flag']:
            reasons = []
            if (res.get('contra_doc_then', 0.0) or 0.0) > 0.0:
                reasons.append("Then-NLI")
            if (res.get('contra_then_neg', 0.0) or 0.0) > 0.0:
                reasons.append("Then-NEG")
            if (res.get('contra_gw_div', 0.0) or 0.0) > 0.0:
                reasons.append("GW-divergence")
            if (res.get('contra_title_div', 0.0) or 0.0) > 0.0:
                reasons.append("Title-divergence")
            if (res.get('contra_adj_div', 0.0) or 0.0) > 0.0:
                reasons.append("Adjacent-divergence")
            if reasons:
                contra_text += f" (источники: {', '.join(reasons)})"
            # print example pairs for structural divergences
            try:
                gw_pairs = det.get('contra_gw_pairs') or det.get('gw_pairs') or []
                title_pairs = det.get('contra_title_pairs') or det.get('title_pairs') or []
                adj_pairs = det.get('contra_adj_pairs') or det.get('adj_pairs') or []
                if gw_pairs:
                    print("  GW-div examples (похожие GW — разные Then):")
                    for p in gw_pairs[:2]:
                        print(f"    · {p.get('a_title','')} ↔ {p.get('b_title','')}")
                        print(f"      GW A: {p.get('a_gw','')[:140]}")
                        print(f"      GW B: {p.get('b_gw','')[:140]}")
                        print(f"      Then A: {p.get('a_then',[])[:3]}")
                        print(f"      Then B: {p.get('b_then',[])[:3]}")
                if title_pairs:
                    print("  Title-div examples (похожие заголовки — разные Then):")
                    for p in title_pairs[:2]:
                        print(f"    · {p.get('a_title','')} ↔ {p.get('b_title','')}; Then A: {p.get('a_then',[])[:3]}; Then B: {p.get('b_then',[])[:3]}")
                if adj_pairs:
                    print("  Adj-div examples (соседние сценарии — разные Then):")
                    for p in adj_pairs[:2]:
                        print(f"    · {p.get('a_title','')} ↔ {p.get('b_title','')}")
                        print(f"      Then A: {p.get('a_then',[])[:3]}; Then B: {p.get('b_then',[])[:3]}")
            except Exception:
                pass
        # Собираем только ненулевые сигналы
        sig_map = [
            ("MD", res.get('feature_contradiction', 0.0)),
            ("MD_NLI", res.get('md_nli_contra', 0.0)),
            ("Then-NLI", res.get('contra_doc_then', 0.0)),
            ("Then-NEG", res.get('contra_then_neg', 0.0)),
            ("GW-div", res.get('contra_gw_div', 0.0)),
            ("Title-div", res.get('contra_title_div', 0.0)),
            ("Adj-div", res.get('contra_adj_div', 0.0)),
        ]
        non_zero = [(k, v) for (k, v) in sig_map if (v or 0.0) > 0.0]
        signals_str = ", ".join(f"{k}={v:.2f}" for k, v in non_zero) if non_zero else "нет"
        md_mode = "учёт .md: включен" if md_enforce else "учёт .md: выключен"
        branch = det.get('semantic_branch', '')
        branch_str = f"; semantic={'kw' if branch=='kw' else ('nli' if branch=='nli' else ('step' if branch=='step' else ('kw_relax' if branch=='kw_relax' else 'none')))}" if branch != '' else ""
        cov_str = f"Покрытие: hit {res['hit_rate']*100.0:.1f}%, ключевые {res['kw_hit_rate']*100.0:.1f}%, goal_sim {res.get('goal_sim', 0.0)*100.0:.1f}% ({'ok' if res.get('goal_hit') else 'low'}){branch_str}"
        print(f"  {passed_str}. {expect_str}. {contra_text}. Сигналы: {signals_str} ({md_mode}). {cov_str}.\n{mi_str}")
        try:
            mi_examples = (res.get('multi_intent') or {}).get('examples') or []
            if mi_examples:
                for ex in mi_examples[:2]:
                    print(f"    · пример лишнего кластера: {ex}")
        except Exception:
            pass
        # Extra explainability: show bottom-3 goal alignment pairs when goal_sim is low
        try:
            if not res.get('goal_hit') and det.get('goal_worst'):
                print("  goal_sim: 3 наименее совпадающих описания 'Функция:' ↔ 'Цель:' (по возрастанию):")
                for desc, s in det['goal_worst']:
                    print(f"    - sim={s*100.0:.1f}%: {desc}")
        except Exception:
            pass

        # Детализированный блок по упавшим метрикам сразу (если % метрик < 100%)
        try:
            metrics_list = det.get('metrics') or []
            enabled_metrics = [m for m in metrics_list if m.get('enabled', True)] or metrics_list
            total_m = len(enabled_metrics) if enabled_metrics else len(metrics_list)
            passed_m = len([m for m in enabled_metrics if bool(m.get('status', False))]) if enabled_metrics else len([m for m in metrics_list if bool(m.get('status', False))])
            percent = int(round(100.0 * (passed_m / max(1, total_m)))) if total_m else 100
            if percent < 100:
                print(f"\n# {res['name']}")
                print(f"% метрик: {percent}%")
                # docs
                if res.get('doc'):
                    print("## docs")
                    print(res['doc'])
                # features
                if res.get('files'):
                    print("## features")
                    for f in res['files']:
                        print(f"- {f}")
                # metrics (failed only)
                print("## metrics (только непройденные)")
                # Group by category
                def _metric_category(metric_name: str) -> str:
                    mn = metric_name.lower()
                    if mn in ("score",):
                        return "scoring"
                    coverage_set = {"hit_rate","kw_hit_rate","nli_hit_rate","step_hit_rate","bm25okapi","tfidf_cosine","jaccard_lemma_overlap","cardinality_alignment","graph_coverage"}
                    per_scen_set = {"per_scenario_alignment","trace_density"}
                    quality_set = {"goal_sim","feature_header_valid","duplicate_scenarios_ratio","contradictions"}
                    feature_def_set = {"terminology_consistency","fvi_mit"}
                    if mn in coverage_set:
                        return "coverage"
                    if mn in per_scen_set:
                        return "per_scenario"
                    if mn in quality_set:
                        return "quality_structure"
                    if mn in feature_def_set:
                        return "feature_definition"
                    return "other"
                failed_by_cat: Dict[str, List[Dict[str, Any]]] = {}
                for m in metrics_list:
                    if bool(m.get('status', True)):
                        continue
                    cat = _metric_category(str(m.get('name','')))
                    failed_by_cat.setdefault(cat, []).append(m)
                for cat, arr in failed_by_cat.items():
                    print(f"### {cat}")
                    for m in arr:
                        mname = str(m.get('name',''))
                        print(f"#### {mname}:")
                        # Concrete fixes with file:line when possible
                        fixes_printed = 0
                        # hit_rate: use misses and best matches
                        if mname == 'hit_rate':
                            misses = det.get('misses_idx') or []
                            req_units = det.get('req_units') or []
                            best_texts = det.get('best_match_texts') or []
                            for ord_idx, i in enumerate(misses[:5], start=1):
                                if 0 <= i < len(req_units):
                                    bt = best_texts[i] if i < len(best_texts) else ''
                                    locs = _find_locations(bt, res.get('files') or []) if bt else []
                                    where = (locs[0] if locs else (res['files'][0] + ':<search>')) if res.get('files') else '<feature>'
                                    print(f"- {ord_idx}: Добавьте Then/And, отражающий требование: \"{req_units[i]}\"; либо скорректируйте шаг \"{bt}\" в {where} (добавьте ключевые термины).")
                                    fixes_printed += 1
                        elif mname == 'kw_hit_rate':
                            # Общая рекомендация (один раз для блока метрики)
                            print("- Общая рекомендация: перенесите ключевые термины из .md в Then/And (те же сущности/атрибуты/направление); избегайте синонимов. Нужны ≥2 общих токена.")
                            kmiss = det.get('kw_misses_idx') or []
                            req_units = det.get('req_units') or []
                            best_idx = det.get('best_match_indices') or []
                            best_txt = det.get('best_match_texts') or []
                            # локальная токенизация в стиле kw_hit_rate
                            import re as _re_tok
                            def _toks(s: str) -> set[str]:
                                return set(_re_tok.findall(r"[\w\-/]+", (s or "").lower()))
                            for ord_idx, i in enumerate(kmiss[:7], start=1):
                                if 0 <= i < len(req_units):
                                    req_text = req_units[i]
                                    j = best_idx[i] if i < len(best_idx) else -1
                                    then_text = best_txt[i] if i < len(best_txt) else ''
                                    shared = list(_toks(req_text) & _toks(then_text)) if then_text else []
                                    req_only = list(_toks(req_text) - set(shared))
                                    then_only = list(_toks(then_text) - set(shared)) if then_text else []
                                    locs = _find_locations(then_text, res.get('files') or []) if then_text else []
                                    where = (locs[0] if locs else (res['files'][0] + ':<search>')) if res.get('files') else '<feature>'
                                    # Предложение: добавить недостающие лексемы из req_only (топ-2)
                                    suggest_add = ", ".join(f"\"{t}\"" for t in req_only[:2]) if req_only else "ключевые слова из .md"
                                    print(f"- {ord_idx}: Требование: \"{req_text}\" — best: \"{then_text}\" → shared={shared or []}; req_only={req_only or []}; then_only={then_only or []}")
                                    print(f"  Предложение: скорректируйте шаг в {where}, добавив ≥2 общих токена (например: {suggest_add}).")
                                    fixes_printed += 1
                        elif mname == 'nli_hit_rate':
                            nmiss = det.get('nli_not_entailed_idx') or []
                            req_units = det.get('req_units') or []
                            worst_pairs = det.get('nli_worst_pairs') or []
                            for ord_idx, i in enumerate(nmiss[:5], start=1):
                                if 0 <= i < len(req_units):
                                    wp = next(((idx, then_text, prob) for (idx, then_text, prob) in worst_pairs if idx == i), None)
                                    then_text = wp[1] if wp else ''
                                    locs = _find_locations(then_text, res.get('files') or []) if then_text else []
                                    where = (locs[0] if locs else (res['files'][0] + ':<search>')) if res.get('files') else '<feature>'
                                    print(f"- {ord_idx}: Уточните Then для entailment требования: \"{req_units[i]}\" — правьте шаг \"{then_text}\" в {where} (добавьте явный ожидаемый результат/направление).")
                                    fixes_printed += 1
                        elif mname == 'step_hit_rate':
                            smiss = det.get('step_misses_idx') or []
                            req_units = det.get('req_units') or []
                            best_then = det.get('step_best_then') or []
                            for ord_idx, i in enumerate(smiss[:5], start=1):
                                if 0 <= i < len(req_units):
                                    bt = next((b[1] for b in best_then if b and b[0] == i), '')
                                    locs = _find_locations(bt, res.get('files') or []) if bt else []
                                    where = (locs[0] if locs else (res['files'][0] + ':<search>')) if res.get('files') else '<feature>'
                                    print(f"- {ord_idx}: Добавьте конкретный Then с ключевыми словами для \"{req_units[i]}\"; можно править \"{bt}\" в {where} (добавьте ≥2 общих токена с .md).")
                                    fixes_printed += 1
                        if fixes_printed == 0:
                            print("- Метрика не содержит конкретных правок, см. раздел общего результата")
        except Exception:
            pass

        # Explain coverage misses with concrete requirement texts
        try:
            req_units = det.get('req_units') or []
            # HIT misses (embedding threshold)
            miss = det.get('misses_idx') or []
            if miss:
                print("  hit_rate: непокрытые требования (нет семантически близкого шага, SIM_THRESHOLD):")
                for ord_idx, i in enumerate(miss[:10], start=1):
                    if 0 <= i < len(req_units):
                        # show best embedding match with cosine and text
                        try:
                            best_idx = det.get('best_match_indices') or []
                            best_sims = det.get('best_match_sims') or []
                            best_texts = det.get('best_match_texts') or []
                            j = best_idx[i] if i < len(best_idx) else -1
                            sim = best_sims[i] if i < len(best_sims) else 0.0
                            bt = best_texts[i] if i < len(best_texts) else ""
                            if j >= 0 and bt:
                                print(f"    - [{ord_idx}] {req_units[i]} — best match: cos={sim:.2f} → \"{bt}\"")
                            else:
                                print(f"    - [{ord_idx}] {req_units[i]} — best match: <none>")
                        except Exception:
                            print(f"    - [{ord_idx}] {req_units[i]}")
                    else:
                        print(f"    - [{ord_idx}] <out of range>")
            # KW misses (token overlap gate)
            kmiss = det.get('kw_misses_idx') or []
            if kmiss:
                print("  kw_hit_rate: требования без ≥2 общих токенов с лучшим совпадением (лечится лексикой):")
                for ord_idx, i in enumerate(kmiss[:10], start=1):
                    if 0 <= i < len(req_units):
                        print(f"    - [{ord_idx}] {req_units[i]}")
                    else:
                        print(f"    - [{ord_idx}] <out of range>")
            # NLI not entailed (print worst Then and top-3 Best Then for each miss)
            nmiss = det.get('nli_not_entailed_idx') or []
            if nmiss:
                try:
                    worst_pairs = det.get('nli_worst_pairs') or []
                    best_top3 = det.get('nli_best_top3_pairs') or []
                    nli_thr = float(det.get('nli_entail_threshold', os.environ.get('NLI_HIT_ENTAIL', '0.70')))
                except Exception:
                    worst_pairs = []
                    best_top3 = []
                    try:
                        nli_thr = float(os.environ.get('NLI_HIT_ENTAIL', '0.70'))
                    except Exception:
                        nli_thr = 0.70
                print("  nli_hit_rate: требования без достаточного entailment от Then/And (порог NLI) — провоцирующие (худшие) Then и top-3 лучших кандидата для улучшения:")
                for ord_idx, i in enumerate(nmiss[:10], start=1):
                    if 0 <= i < len(req_units):
                        # find worst pair for this requirement index
                        wp = next(((idx, then_text, prob) for (idx, then_text, prob) in worst_pairs if idx == i), None)
                        if wp is not None:
                            _idx, then_text, prob = wp
                            print(f"    - [{ord_idx}] {req_units[i]} — worst Then: \"{then_text}\", entailment={prob:.2f} (<{nli_thr:.2f})")
                        else:
                            print(f"    - [{ord_idx}] {req_units[i]} — worst Then: <not found>")
                        # print top-3 best Then suggestions to tweak phrasing
                        tops = [(idx, t, p) for (idx, t, p) in best_top3 if idx == i][:3]
                        if tops:
                            for (_idx2, t2, p2) in tops:
                                print(f"        · best: \"{t2}\" (entailment={p2:.2f})")
                        else:
                            print("        · best: <no candidates>")
                    else:
                        print(f"    - [{ord_idx}] <out of range>")
            # STEP misses
            smiss = det.get('step_misses_idx') or []
            if smiss:
                print("  step_hit_rate: требования без лексического перекрытия с Then/And (≥2 общих токена):")
                for ord_idx, i in enumerate(smiss[:10], start=1):
                    if 0 <= i < len(req_units):
                        try:
                            overlaps = det.get('step_token_overlap') or []
                            best_then = det.get('step_best_then') or []
                            # find by index
                            ov = next((o for o in overlaps if o and o[0] == i), None)
                            bt = next((b for b in best_then if b and b[0] == i), None)
                            if ov and bt:
                                _idx, shared, req_only, then_only = ov
                                _idx2, then_text, inter = bt
                                print(f"    - [{ord_idx}] {req_units[i]} — best Then: \"{then_text}\" (overlap={inter}); shared={shared or []}; req_only={req_only or []}; then_only={then_only or []}")
                            else:
                                print(f"    - [{ord_idx}] {req_units[i]}")
                        except Exception:
                            print(f"    - [{ord_idx}] {req_units[i]}")
                    else:
                        print(f"    - [{ord_idx}] <out of range>")
        except Exception:
            pass
        # Разделитель между результатами по разным парам .md ↔ .feature
        print("")
    # General summary: reuse unexpected outcomes list to compute correctness
    total = len(results)
    if total > 0:
        correct = total - len(failed)
        # Всегда печатаем сводку по количеству фич и сценарием
        print(f"\nИтого: фич={total}, сценариев={scenarios_total}")
        if correct == total:
            print(f"[OK] Проверка алгоритма: корректно {correct} из {total} фич")
        else:
            print(f"[!] Проверка алгоритма: корректно {correct} из {total} фич")

        if failed:
            print(f"\nПРОВАЛ (покрытие ниже порога/или неверное ожидание):")

            def _metric_category(metric_name: str) -> str:
                mn = metric_name.lower()
                if mn in ("score",):
                    return "scoring"
                coverage_set = {"hit_rate","kw_hit_rate","nli_hit_rate","step_hit_rate","bm25okapi","tfidf_cosine","jaccard_lemma_overlap","cardinality_alignment","graph_coverage"}
                per_scen_set = {"per_scenario_alignment","trace_density"}
                quality_set = {"goal_sim","feature_header_valid","duplicate_scenarios_ratio","contradictions"}
                feature_def_set = {"terminology_consistency","fvi_mit"}
                if mn in coverage_set:
                    return "coverage"
                if mn in per_scen_set:
                    return "per_scenario"
                if mn in quality_set:
                    return "quality_structure"
                if mn in feature_def_set:
                    return "feature_definition"
                return "other"

            for res in results:
                name = res['name']
                expected_fail_res = bool(res.get('expected_fail', False))
                # Только для действительно непройденных (unexpected)
                if is_unexpected_outcome(expected_fail=expected_fail_res, raw_passed=bool(res.get('raw_passed', False))):
                    det = res.get('details', {}) or {}
                    metrics = det.get('metrics') or []
                    # Сбор непройденных метрик
                    failed_by_cat: dict[str, list[str]] = {}
                    for m in metrics:
                        try:
                            if not bool(m.get('status', False)):
                                cat = _metric_category(str(m.get('name','')))
                                val = float(m.get('value', 0.0))
                                # формат значения: целое для процентов score, иначе 2 знака
                                if str(m.get('name','')).lower() == 'score':
                                    val_str = f"{val:.0f}%"
                                else:
                                    val_str = f"{val:.2f}"
                                failed_by_cat.setdefault(cat, []).append(f"{m.get('name')}={val_str}")
                        except Exception:
                            continue
                    if failed_by_cat:
                        parts = []
                        for cat, items in failed_by_cat.items():
                            parts.append(f"{cat}: {', '.join(items)}")
                        line = f" - {name}: " + "; ".join(parts)
                        print(line)
                    else:
                        # Fallback на старые причины, если метрик нет
                        reasons = _collect_fail_reasons(res, threshold=threshold, threshold_pct=threshold_pct)
                        if strict_should_fail and expected_fail_res:
                            print(f" - {name}: Ожидается неудача (strict @should-fail), но фича прошла: " + "; ".join(reasons))
                        else:
                            print(f" - {name}: " + "; ".join(reasons))

            print("\n[X] Проверка покрытия требований не пройдена!")
            print("[?] Подробности см. в отчёте анализатора выше для выбранной категории")
            # Пояснение (глоссарий) выводим только при наличии провалов
            print("\nПояснение (что не так, где смотреть, как чинить):")
            # Основные скоры
            print("- score — интегральная близость .md ↔ .feature (0–100%).")
            print("  Проблема: низкий score = тексты в целом про разное.")
            print("  Где смотреть: строка ‘name: X%’, ‘model_cosine’, тексты .md/.feature выше.")
            print("  Как чинить: согласовать терминологию и смысл: переформулировать ‘Цель:’ и ‘Функция:’, добавить ключевые шаги/Then с теми же сущностями.")
            print("- agg_base — базовое сходство (косинус без бонусов).")
            print("  Проблема: низкий agg_base = тексты разнонаправленные; высокий agg_base смягчает требование к kw_hit_rate.")
            print("  Где смотреть: строка результата ‘agg_base: X%’ (и ‘model_cosine’).")
            print("  Как чинить: уточнить .md, убрать лишние темы или разделить фичи; привести названия и роли к тем, что проверяются шагами.")
            # Покрытие по требованиям (по каждому — проблема/где/как)
            print("- hit_rate — есть ли семантически близкий шаг для каждого пункта .md (порог SIM_THRESHOLD; минимум MIN_HIT_RATE/min_hit_rate_dynamic).")
            print("  Проблема: часть требований .md не находит близких шагов.")
            print("  Где смотреть: ‘hit_rate: непокрытые требования …’ (список req_units по индексам); код: tests/analyze/metrics/hit_rate.py (misses_idx, matches).")
            print("  Как чинить: добавить Then/And, дословно фиксирующие ожидаемый результат требования; либо перефразировать .md/шаги, чтобы смысл совпал.")
            print("- kw_hit_rate — лексическое перекрытие (≥2 общих токена с лучшим совпадением).")
            print("  Проблема: шаг формально близок, но написан другими словами.")
            print("  Где смотреть: ‘kw_hit_rate: требования без ≥2 общих токенов …’; код: tests/analyze/metrics/kw_hit_rate.py (kw_misses_idx, token_intersections).")
            print("  Как чинить: использовать в шагах те же термины из .md (сущности, атрибуты, направление изменения), избегать синонимов.")
            print("- nli_hit_rate — логическое покрытие Then/And по NLI (entailment).")
            print("  Проблема: Then/And не подтверждают формулировку требования (низкий entailment).")
            print("  Где смотреть: ‘nli_hit_rate: требования без достаточного entailment …’; код: tests/analyze/metrics/nli_hit_rate.py (best_pairs, not_entailed_idx).")
            print("  Как чинить: добавить явные Then/And про ожидаемый результат, направление и величину эффекта; уточнить Given/When для однозначности.")
            print("- step_hit_rate — структурное покрытие: есть ли Then/And с ≥2 общими токенами с требованием .md.")
            print("  Проблема: шаги слишком общие/косвенные, нет пересечения по словам.")
            print("  Где смотреть: ‘step_hit_rate: требования без лексического перекрытия …’; код: tests/analyze/metrics/step_hit_rate.py (step_misses_idx, intersections).")
            print("  Как чинить: добавить конкретные Then/And с теми же сущностями/атрибутами, что в .md; не прятать итог в ‘И’ без ключевых слов.")
            # Условие semantic_coverage
            print("- semantic_coverage — проходит, если: kw_hit_rate ≥ 0.60 ИЛИ nli_hit_rate ≥ 0.60 ИЛИ step_hit_rate ≥ 0.60; при agg_base ≥ 0.82 — достаточно kw_hit_rate ≥ 0.40.")
            print("  Проблема: при нормальном hit_rate прокси‑покрытие по ключам/NLI/структуре ниже порога.")
            print("  Где смотреть: причина в строке ‘semantic_coverage слабое: kw=…, nli=…, step=…’.")
            print("  Как чинить: выбрать быстрый путь — поднять kw_hit_rate согласованием терминов, либо добавить явный Then/And для NLI, либо структурный Then с нужными ключами для step_hit_rate.")
            # Пороговые параметры
            print("- Пороги: THRESHOLD (общий порог score), MIN_HIT_RATE (базовый hit_rate), SIM_THRESHOLD (семантическая близость).")
            print("  Где менять: tests/analyze/config.json или переменные окружения при запуске.")
            # Цель ↔ Функция
            print("- goal_sim / goal_hit — ‘Цель:’ ↔ конкатенация всех ‘Функция:’.")
            print("  Проблема: ‘Функция:’ описывает другое, чем ‘Цель:’ (низкий goal_sim).")
            print("  Где смотреть: ‘goal_sim … (low)’ и ‘3 наименее совпадающих описания ‘Функция:’ …’. Код: compute_goal_alignment в tests/analyze/metrics/coverage.py.")
            print("  Как чинить: перефразировать заголовки ‘Функция:’ так, чтобы они прямо поддерживали ‘Цель:’; убрать лишние/косвенные функции.")
            # Multi-intent
            print("- multi_intent — признак нескольких тем в .md.")
            print("  Проблема: смесь разных целей/ценностей снижает согласованность шагов и метрики.")
            print("  Где смотреть: строка ‘multi_intent: True …’ в результате.")
            print("  Как чинить: сузить ‘Цель:’ и описание до одной поведенческой цели или разделить на несколько фич.")
            # Противоречия
            print("- contradiction — структурные/NLI‑противоречия.")
            print("  Then‑NLI: Then друг другу противоречат по NLI → объединить/уточнить ожидаемые исходы.")
            print("  Then‑NEG: очень похожие Then с разной полярностью → убрать дубли/противоречия, развести предусловия.")
            print("  GW‑div: Given/When похожи, Then расходятся → добавить различающие условия в Given/When или объединить в Scenario Outline.")
            print("  Title‑div: похожие заголовки со разными исходами → переименовать заголовки или унифицировать исходы.")
            print("  Adj‑div: соседние сценарии с почти одинаковыми GW, но разными Then → слить в Outline или сделать условия различимыми.")
            # Should-fail
            print("- should‑fail — отрицательные примеры: ожидается провал; при STRICT_SHOULD_FAIL=1 засчитываются как успешно пройденные негативные (без причин в сводке).")
            # Где смотреть реализацию метрик
            print("\nГде смотреть реализацию и сигналы в коде:")
            print("- hit_rate: tests/analyze/metrics/hit_rate.py (misses_idx, matches)")
            print("- kw_hit_rate: tests/analyze/metrics/kw_hit_rate.py (kw_misses_idx, token_intersections)")
            print("- nli_hit_rate: tests/analyze/metrics/nli_hit_rate.py (best_pairs, not_entailed_idx)")
            print("- step_hit_rate: tests/analyze/metrics/step_hit_rate.py (step_misses_idx, intersections)")
            print("- goal_sim: tests/analyze/metrics/coverage.py → compute_goal_alignment (печать bottom‑3)")
            print("- противоречия: tests/analyze/metrics/contradictions.py, tests/analyze/metrics/nli_then.py")
            print("- интегральная логика/пороги: tests/analyze/metrics/evaluation.py")
            print("\n[!] Быстрые действия (по симптомам):")
            print("  • Низкие score/agg_base: узко переформулировать ‘Цель:’/‘Функция:’, удалить лишнее, при необходимости разделить фичу.")
            print("  • Низкий hit_rate: добавить недостающие Then/And, явно фиксирующие ожидаемый результат для каждого пункта .md.")
            print("  • Низкий kw_hit_rate: согласовать лексику — использовать в шагах те же термины/сущности/атрибуты, что в .md.")
            print("  • Низкий nli_hit_rate: уточнить Given/When и добавить Then/And с явным направлением/величиной результата.")
            print("  • Низкий step_hit_rate: сделать шаги конкретными, с ключевыми словами из требований; не прятать результат в общих формулировках.")
            print("  • ‘semantic_coverage слабое’: выбрать быстрый рычаг — поднять kw (лексика), либо nli (явный Then/And), либо step (структурный Then).")
            print("  • Низкий goal_sim: переписать ‘Функция:’ под ‘Цель:’, убрать несогласованные блоки.")
            print("  • Противоречия (GW/Title/Adj/Then-NEG/NLI): объединить дубли, различить предусловия, перевести в Scenario Outline с Examples или унифицировать исходы.")
            print("  • Много намерений (multi_intent=True): сузить описание или разделить на отдельные фичи.")
            print("  • Пороговая настройка: при временной диагностике можно ослабить THRESHOLD/MIN_HIT_RATE либо запустить со SKIP_REQUIREMENTS_CHECK=1.")
            # Дополнительно: формируем секции только по упавшим метрикам
            try:
                from metrics import kw_hit_rate as _kw, hit_rate as _hr, nli_hit_rate as _nli, step_hit_rate as _step
                from metrics import contradictions as _contra
                metric_modules = {
                    "hit_rate": _hr,
                    "kw_hit_rate": _kw,
                    "nli_hit_rate": _nli,
                    "step_hit_rate": _step,
                    "contradictions": _contra,
                }
            except Exception:
                metric_modules = {}

            # удалено: детализированный секционный блок — все детали уже печатаются выше

            raise SystemExit(1)
    else:
        # Даже если ничего не отфильтровалось, печатаем сводку с нулями
        print(f"\nИтого: фич=0, сценариев={scenarios_total}")
        print("[OK] Проверка алгоритма: корректно 0 из 0 фич")


if __name__ == "__main__":
    main()
