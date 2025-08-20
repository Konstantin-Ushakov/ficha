## Категория 5 — Дополнительные (качество/структура)

См. также: `../assistant.md` (раздел «5) Дополнительные (качество/структура)»).

### Валидность шапки `.feature`
- Что проверяет: корректность второй строки с тегами `@<имя_фичи>` и `@<категория>`.
- Опора в коде: `tests/analyze/metrics/extras.py` → `feature_header_valid` (включается `FEATURE_HEADER_VALID_ENABLE`).
- Формула: `1.0`, если на второй непустой строке присутствуют теги и среди них есть `@<число>`; иначе `0.0`.

### Дубликаты сценариев
- Что проверяет: долю пар `Сценарий:` с высокой лексической близостью заголовков (Jaccard по токенам).
- Опора в коде: `tests/analyze/metrics/extras.py` → `duplicate_scenarios_ratio` (включается `DUP_SCENARIOS_ENABLE`).
- Формула: `dup_pairs / all_pairs`, где пара считается дубликатом при `Jaccard ≥ DUP_SCENARIO_JACC` (по умолчанию `0.8`).

### Goal≈Function (GFS)
- Что проверяет: близость `Цель:` из .md к описаниям `Функция:` в `.feature`.
- Опора в коде: `tests/analyze/metrics/coverage.py` → `compute_goal_alignment` (включается `GOAL_ENABLE`).
- Формула: косинусная близость эмбеддингов `cos(emb(goal), emb(concat(Функция:)))` ∈ [0,1]; `goal_hit = goal_sim ≥ GOAL_SIM_THRESHOLD`.

### Противоречия (Then‑NLI, Then‑NEG, GW/Title/Adj‑div)
- Что проверяет: логические и структурные противоречия внутри `.feature`.
- Опора в коде: `tests/analyze/metrics/contradictions.py` и смежные модули (включается `CONTRA_ENABLE`).
- Формулы/эвристики:
  - Then‑NEG: похожие Then с противоположной полярностью → детект через токен‑пересечение и отрицания.
  - Then‑NLI: противоречие по NLI между Then → модель NLI ≥ порог.
  - GW/Title/Adj‑div: близкие Given/When/заголовки с расходящимися Then → порог по схожести GW/заголовка, Jaccard по Then, пороги `GW_SIM_ADJ`, `ADJ_THEN_JACC_MAX` и пр.



