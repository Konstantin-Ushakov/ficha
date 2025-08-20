## Категория 3 — Покрыты ли все сценарии .md в .feature

См. также: `../assistant.md` (раздел «3) Покрыты ли все сценарии .md в .feature»).

### kw_hit_rate
- Что проверяет: лексическое перекрытие требований .md с лучшими совпадениями шагов `.feature` (≥2 общих токена).
- Опора в коде: `tests/analyze/metrics/kw_hit_rate.py` и агрегация в `coverage.compute_coverage`.
- Формула: `kw_hit_rate = (# требований, где |tokens(r_i)∩tokens(best_u_i)|≥2) / |R|`.

### Jaccard lemma overlap
- Что проверяет: средняя похожесть по множествам токенов между требованием и лучшим Then/юнитом.
- Опора в коде: `tests/analyze/metrics/extras.py` → `jaccard_lemma_overlap` (включается `JACCARD_LEMMA_ENABLE`).
- Формула: `J = |A∩B|/|A∪B|`, усреднение по всем `r_i`.

### Cardinality alignment
- Что проверяет: совпадение числовых ожиданий (кардинальности) между .md и Then.
- Опора в коде: `tests/analyze/metrics/extras.py` → `cardinality_alignment` (включается `CARDINALITY_ALIGNMENT_ENABLE`).
- Формула: для каждого `r_i` сравнить множества чисел `nums(r_i)` и `nums(best_u_i)`; метрика — доля точных совпадений (пустое множество считается нейтральным успехом).

### BM25Okapi (feature coverage proxy)
- Что проверяет: IR‑похожесть требований к шагам `.feature` как прокси‑покрытие.
- Опора в коде: `tests/analyze/metrics/extras.py` → `bm25okapi` (включается `BM25_ENABLE`).
- Формула: среднее `bm25_best(r_i, Docs)` по всем `r_i`, где `Docs` — токенизированные юниты `.feature`.

### step_hit_rate
- Что проверяет: структурное покрытие Then/And (≥2 общих токена) против требований .md.
- Опора в коде: `tests/analyze/metrics/step_hit_rate.py` и агрегация в `coverage.compute_coverage`.
- Формула: доля требований, для которых найден Then/And с ≥2 общими токенами.

### TF‑IDF cosine (feature coverage proxy)
- Что проверяет: TF‑IDF косинусная близость требование→юниты `.feature`.
- Опора в коде: `tests/analyze/metrics/extras.py` → `tfidf_cosine` (включается `TFIDF_ENABLE`).
- Формула: средний максимум по косинусу для каждого `r_i`.

### Graph coverage analysis
- Что проверяет: разнообразие трасс (уникальность лучших попаданий) — нет ли «схлопывания» многих требований в один шаг.
- Опора в коде: `tests/analyze/metrics/extras.py` → `graph_coverage` (включается `GRAPH_COVERAGE_ENABLE`).
- Формула: `graph_coverage = |unique(best_match_indices)| / |R|` ∈ [0,1].



