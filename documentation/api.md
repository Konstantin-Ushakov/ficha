# API документация FiCha

## Основные модули

### analyzer.py
Главный модуль анализатора, содержащий CLI интерфейс и основную логику.

#### Основные функции:
- `main()` - точка входа CLI
- `analyze_features()` - основной анализ фич
- `load_config()` - загрузка конфигурации

### core/
Основные утилиты и модели.

#### models.py
- `load_embedding_model()` - загрузка модели эмбеддингов
- `load_nli_model()` - загрузка NLI модели
- `compute_embeddings()` - вычисление эмбеддингов

#### preprocessing.py
- `split_markdown()` - разбиение markdown на смысловые единицы
- `tokenize_text()` - токенизация текста
- `clean_text()` - очистка текста

#### scoring.py
- `compute_cosine_similarity()` - косинусное сходство
- `compute_jaccard_similarity()` - Jaccard сходство
- `soft_f1_score()` - мягкий F1 score

### metrics/
Реализация всех метрик анализа.

#### coverage.py
- `compute_coverage()` - основная функция покрытия
- `compute_goal_alignment()` - выравнивание целей
- `hit_rate_metric()` - метрика hit_rate

#### contradictions.py
- `detect_contradictions()` - детект противоречий
- `then_neg_contradictions()` - противоречия в Then
- `adjacent_contradictions()` - противоречия в соседних сценариях

#### nli_then.py
- `nli_then_analysis()` - NLI анализ Then шагов
- `compute_entailment()` - вычисление entailment

## Конфигурация

### Основные параметры:
- `EMBEDDING_MODEL` - модель для эмбеддингов
- `NLI_MODEL_DIR` - путь к NLI модели
- `SIM_THRESHOLD` - порог семантического сходства
- `GOAL_SIM_THRESHOLD` - порог для goal alignment

### Метрики:
Каждая метрика может быть включена/выключена через `ENABLE` флаги в соответствующей секции.

## Использование

### CLI интерфейс:
```bash
python analyzer.py [--config CONFIG] [--category CATEGORY]
```

### Программный интерфейс:
```python
from analyzer import analyze_features
from config import load_config

config = load_config("config.json")
results = analyze_features(config, category="@demo")
```

## Возвращаемые данные

Анализатор возвращает словарь с результатами по каждой фиче:
- `score` - общий score
- `hit_rate` - покрытие требований
- `kw_hit_rate` - лексическое покрытие
- `goal_sim` - сходство целей
- `contradictions` - найденные противоречия
- `recommendations` - рекомендации по улучшению

