# Feature Coverage Analyzer (FiCha)

Семантический анализатор покрытия требований фич для проектов BDD/TDD, использующих Cucumber.

> Разработка с использованием AI: проект создан с поддержкой AI‑инструментов (генерация кода/документации, оптимизации, тестирование). См. подробности в `documentation/assistant.md`.

## Зачем нужен FiCha?

FiCha автоматически анализирует соответствие между описаниями фич в markdown-файлах и их реализацией в Cucumber feature-тестах, используя машинное обучение для семантического сравнения. Инструмент выявляет непокрытые требования, лексические несоответствия, семантические пробелы и структурные проблемы в тестах. Это помогает разработчикам и QA-инженерам поддерживать качество документации, улучшать покрытие тестами и выявлять противоречия на ранних этапах разработки.

## Назначение

FiCha проверяет корректность фич (`.md`), фиче-тестов (`.feature`) и соответствие между ними:

- **Валидация фич**: проверка, что `.md` документ соответствует определению фичи (структура, цели, требования)
- **Валидация тестов**: проверка корректности Cucumber `.feature` файлов (формат, структура, отсутствие противоречий)
- **Покрытие требований**: соответствие всех требований из `.md` шагам Cucumber тестов
- **Goal alignment**: согласованность целей фичи с описанием функций в тестах
- **Качество структуры**: выявление дубликатов, противоречий и структурных проблем

## Быстрый старт

### 🚀 Демо (рекомендуется для первого знакомства)
```bash
# Находимся в папке tests/analyze
cd demo && ./demo.sh
```

Демо покажет, как работает FiCha на простом примере и поможет понять интерпретацию результатов.

### Локальный запуск без внешних скриптов
```bash
# Находимся в папке tests/analyze
# Анализ с категорией по тегу
TEST_CATEGORY=10 docker compose -f docker-compose.yml run --rm semantic-analyzer

# Анализ до категории X включительно (пример: till-3)
TEST_CATEGORY=till-3 docker compose -f docker-compose.yml run --rm semantic-analyzer
```

### Прямой запуск через Docker Compose (без docker-compose/XXX)
```bash
# Находимся в папке tests/analyze
docker compose -f docker-compose.yml run --rm semantic-analyzer

# С переменными окружения
TEST_CATEGORY=10 docker compose -f docker-compose.yml run --rm semantic-analyzer
```

## Архитектура

```
tests/analyze/
├── core/           # Основные модули анализа
├── metrics/        # Реализация метрик
├── model/          # Оффлайн-модели NLI и эмбеддингов
├── documentation/  # Детальная документация метрик
├── docker-compose.yml  # Конфигурация сервиса
└── env.example    # Примеры переменных окружения
```

## Метрики и пороги (кратко)

Одна ключевая метрика для каждой категории. Числа берутся из `config.json`.

- Feature Definition: `FVI_mit` — валидность `.md` как фичи (см. `documentation/metrics/md/1_feature_definition.md`)
- MD Duplicates: `BM25Okapi` — поиск близких `.md` (см. `documentation/metrics/md/2_md_duplicates.md`)
- Coverage: `hit_rate` — общее покрытие требований; рекомендуемый порог: `COV_SCORE_HIT=0.80` (см. `config.json`, а также `SIM_THRESHOLD=0.45`)
- Per Scenario: `per_scenario_alignment` — соответствие сценариев аспектам (см. `documentation/metrics/md/4_per_scenario.md`)
- Quality Structure: `goal_alignment` — согласованность `Цель:` ↔ `Функция:`; порог: `GOAL_SIM_THRESHOLD=0.60`

## Конфигурация

### Переменные окружения
Скопируйте `env.example` в `.env` и настройте. Ключевые пороги и параметры — в `config.json`.

### Пороги метрик
Все актуальные значения берутся из `config.json` (подробные комментарии — в `config.example.jsonc`).

## Документация

### Основные документы
- [`documentation/assistant.md`](./documentation/assistant.md) - полный обзор метрик и критериев
- [`documentation/output_format.md`](./documentation/output_format.md) - формат выходных данных
- [`documentation/api.md`](./documentation/api.md) - API документация и использование
- [`documentation/troubleshooting.md`](./documentation/troubleshooting.md) - решение проблем и отладка

### Детальные описания метрик
- [`documentation/metrics/md/1_feature_definition.md`](./documentation/metrics/md/1_feature_definition.md) - определение фич
- [`documentation/metrics/md/2_md_duplicates.md`](./documentation/metrics/md/2_md_duplicates.md) - дублирование в markdown
- [`documentation/metrics/md/3_coverage.md`](./documentation/metrics/md/3_coverage.md) - покрытие сценариями
- [`documentation/metrics/md/4_per_scenario.md`](./documentation/metrics/md/4_per_scenario.md) - анализ по сценариям
- [`documentation/metrics/md/5_quality_structure.md`](./documentation/metrics/md/5_quality_structure.md) - качество структуры

## Разработка и настройка

### Добавление новых метрик
1. Создайте модуль в `tests/analyze/metrics/`
2. Реализуйте функцию с сигнатурой `metric_name(features, tests)`
3. Добавьте в `tests/analyze/metrics/__init__.py`
4. Обновите документацию

### Настройка порогов
- Измените константы в соответствующих модулях метрик
- Протестируйте на реальных данных
- Обновите `env.example` при необходимости

### Отладка
```bash
# Проверка конфигурации Docker
docker compose -f docker-compose.yml config
```

## Требования и платформы

- Docker и Docker Compose
- Оффлайн‑модели в `tests/analyze/model/`
- Python 3.8+ (для локальной разработки)
- Поддерживаемые платформы: Windows (Git Bash или WSL), Linux

## Лицензия

См. файл [LICENSE](./LICENSE) для детальной информации о правах использования.

**Проприетарное программное обеспечение** - все права защищены.
