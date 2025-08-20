# Troubleshooting FiCha

## Частые проблемы и их решения

### 1. Ошибки загрузки моделей

#### Проблема: "Model not found"
```
Error: Could not find model at /app/model/...
```

**Решение:**
- Убедитесь, что модели скачаны в папку `tests/analyze/model/`
- Проверьте пути в `config.json` соответствуют реальной структуре
- Запустите `./demo/demo.sh` для проверки работоспособности

#### Проблема: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```

**Решение:**
- Используйте CPU версию: `export CUDA_VISIBLE_DEVICES=""`
- Уменьшите `NLI_TOPK` и `NLI_BOTTOMK` в конфигурации
- Запускайте анализ по одной категории за раз

### 2. Ошибки Docker

#### Проблема: "Service not found"
```
Error: Service 'semantic-analyzer' not found
```

**Решение:**
- Запускайте из папки `tests/analyze`
- Проверьте, что `tests/analyze/docker-compose.yml` доступен
- Команда: `cd tests/analyze && docker compose -f docker-compose.yml run --rm semantic-analyzer`

#### Проблема: "Permission denied"
```
Error: Permission denied on model files
```

**Решение:**
- Проверьте права доступа к папке `model/`
- Убедитесь, что Docker может монтировать volumes

### 3. Проблемы с анализом

#### Проблема: Низкие метрики покрытия
```
hit_rate: 0.30 (ожидалось >= 0.67)
```

**Решение:**
- Проверьте соответствие терминологии в `.md` и `.feature`
- Убедитесь, что все требования разбиты на отдельные пункты
- Проверьте, что каждый пункт покрыт соответствующим Then/И шагом

#### Проблема: Противоречия в тестах
```
Contradictions: Then-NEG detected
```

**Решение:**
- Найдите похожие Then с противоположной полярностью
- Объедините в `Scenario Outline` или явно различайте предусловия
- Убедитесь, что соседние сценарии не расходятся без причины

### 4. Проблемы производительности

#### Проблема: Медленный анализ
```
Analysis took 5+ minutes
```

**Решение:**
- Отключите неиспользуемые метрики в `config.json`
- Уменьшите `NLI_TOPK` и `NLI_BOTTOMK`
- Используйте более легкие модели эмбеддингов

#### Проблема: Высокое потребление памяти
```
Memory usage > 4GB
```

**Решение:**
- Запускайте анализ по частям
- Используйте `--category` для ограничения объема
- Проверьте настройки Docker memory limits

## Логи и отладка

### Включение подробных логов:
```bash
export LOG_LEVEL=DEBUG
./demo/demo.sh
```

### Проверка конфигурации:
```bash
cd tests/analyze && docker compose -f docker-compose.yml config
```

### Тестирование отдельных компонентов:
```bash
cd tests/analyze
python -c "from core.models import load_embedding_model; print('Models OK')"
```

## Получение помощи

1. **Проверьте демо**: `./demo/demo.sh`
2. **Изучите документацию**: `documentation/`
3. **Проверьте конфигурацию**: `config.example.jsonc`
4. **Запустите с минимальными настройками**: отключите сложные метрики

## Отчеты об ошибках

При создании issue укажите:
- Версию FiCha
- Конфигурацию (config.json)
- Полный вывод ошибки
- Шаги для воспроизведения
- Ожидаемое поведение

