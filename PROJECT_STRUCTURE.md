# Структура проекта FiCha

```
tests/analyze/
├── 📁 core/                    # Основные модули
│   ├── models.py              # Загрузка ML моделей
│   ├── preprocessing.py        # Обработка текста
│   ├── scoring.py             # Алгоритмы скоринга
│   └── io_utils.py            # Работа с файлами
│
├── 📁 metrics/                 # Реализация метрик
│   ├── coverage.py            # Покрытие требований
│   ├── contradictions.py      # Детект противоречий
│   ├── nli_then.py           # NLI анализ
│   ├── multi_intent.py       # Многосмысловость
│   └── evaluation.py         # Итоговая оценка
│
├── 📁 model/                   # Оффлайн ML модели
│   ├── sentence-transformers/ # Модели эмбеддингов
│   └── NLI/                   # Модели для NLI
│
├── 📁 documentation/           # Документация
│   ├── assistant.md           # Обзор метрик
│   ├── output_format.md       # Формат вывода
│   ├── api.md                 # API документация
│   ├── troubleshooting.md     # Решение проблем
│   └── 📁 metrics/md/        # Детальные метрики
│       ├── 1_feature_definition.md
│       ├── 2_md_duplicates.md
│       ├── 3_coverage.md
│       ├── 4_per_scenario.md
│       └── 5_quality_structure.md
│
├── 📁 demo/                    # Демонстрационные материалы
│   ├── demo.sh                # Скрипт запуска демо
│   ├── env.demo               # Переменные окружения
│   ├── README.md              # Инструкции по демо
│   └── 📁 example/           # Примеры фич и тестов
│       ├── 0_demo-feature.md
│       └── 0_demo-feature.feature
│
├── analyzer.py                 # Главный модуль
├── parser.py                   # Парсинг файлов
├── config.json                 # Конфигурация
├── config.example.jsonc        # Пример конфигурации с комментариями
├── requirements.txt            # Python зависимости
├── Dockerfile                  # Docker образ
├── docker-compose.yml          # Конфигурация сервиса
├── .dockerignore               # Исключения для Docker
├── .env.example                # Пример переменных окружения
├── LICENSE                     # Проприетарная лицензия
├── README.md                   # Основная документация
└── PROJECT_STRUCTURE.md        # Этот файл
```

## Назначение компонентов

### Core модули
- **models.py** - загрузка и управление ML моделями
- **preprocessing.py** - подготовка текста для анализа
- **scoring.py** - вычисление различных метрик сходства
- **io_utils.py** - чтение файлов и парсинг

### Метрики
- **coverage.py** - основная логика покрытия требований
- **contradictions.py** - детект логических противоречий
- **nli_then.py** - семантический анализ через NLI
- **multi_intent.py** - анализ многосмысловости фич
- **evaluation.py** - итоговая оценка и принятие решений

### Документация
- **assistant.md** - полный обзор всех метрик
- **output_format.md** - описание формата вывода
- **api.md** - программный интерфейс
- **troubleshooting.md** - решение проблем

### Демо
- **demo.sh** - автоматический запуск демо
- **example/** - примеры фич и тестов для обучения

## Запуск

### Демо (рекомендуется для начала)
```bash
cd tests/analyze
./demo/demo.sh
```

### Полный анализ (через локальный compose)
```bash
# Находимся в папке tests/analyze
TEST_CATEGORY=<категория_или_till-X> docker compose -f docker-compose.yml run --rm semantic-analyzer
```

### Docker
```bash
cd tests/analyze
docker compose -f docker-compose.yml run --rm semantic-analyzer
```

