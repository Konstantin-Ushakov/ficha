## Категория 4 — Каждый сценарий проверяет аспект фичи

См. также: `../assistant.md` (раздел «4) Каждый сценарий проверяет аспект фичи»).

### Пер‑сценарное соответствие (per_scenario_alignment)
- Что проверяет: у каждого `Сценарий:` есть хотя бы один Then, лексически пересекающийся с каким‑либо требованием .md (≥2 токена).
- Опора в коде: `tests/analyze/metrics/extras.py` → `per_scenario_alignment` (включается `PER_SCENARIO_ENABLE`).
- Формула: `covered_scenarios / total_scenarios`.

### Trace density
- Что проверяет: «плотность» Then по отношению к количеству требований .md.
- Опора в коде: `tests/analyze/metrics/extras.py` → `trace_density` (включается `TRACE_DENSITY_ENABLE`).
- Формула: `min(1.0, then_count / |R|)`.



