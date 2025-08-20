#!/bin/bash

# Демо-скрипт для FiCha (Feature Coverage Analyzer)
# Запускает семантический анализ на демо-примерах

set -e

echo "🚀 Запуск демо FiCha (Feature Coverage Analyzer)"
echo "=================================================="

# Проверяем, что мы в правильной директории
if [ ! -f "../analyzer.py" ]; then
    echo "❌ Ошибка: Запустите скрипт из папки tests/analyze/demo/"
    echo "   Текущая директория: $(pwd)"
    exit 1
fi

# Проверяем наличие демо-файлов
if [ ! -f "example/0_demo-fail.md" ] || [ ! -f "example/0_demo-fail.feature" ] || [ ! -f "example/0_demo-success.md" ] || [ ! -f "example/0_demo-success.feature" ]; then
    echo "❌ Ошибка: Демо-файлы не найдены в папке example/"
    exit 1
fi

echo "✅ Демо-файлы найдены"
echo "📁 Анализируем:"
echo "   - 0_demo-fail.md ↔ 0_demo-fail.feature (пример с проблемами)"
echo "   - 0_demo-success.md ↔ 0_demo-success.feature (пример успешной фичи)"
echo ""

# Загружаем переменные из env.demo
echo "🔍 Загрузка переменных из env.demo..."
export $(grep -v '^#' env.demo | xargs)

echo "📋 Переменные окружения:"
echo "   USER_FEATURES_PATH: ${USER_FEATURES_PATH}"
echo "   FEATURE_TESTS_PATH: ${FEATURE_TESTS_PATH}"
echo "   MODEL_PATH: ${MODEL_PATH}"
echo "   TEST_CATEGORY: ${TEST_CATEGORY}"
echo ""

# Запускаем анализ через основной Docker Compose с демо-переменными
echo "🔍 Запуск семантического анализа..."
echo "🐳 Запуск через tests/analyze/docker-compose.yml с демо-переменными..."

# Проверяем, есть ли Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не найден. Установите Docker для запуска демо."
    exit 1
fi

# Запускаем анализ с переменными из env.demo
# Переходим в папку tests/analyze для локального compose
cd ..

# Экспортируем переменные для Docker Compose
export USER_FEATURES_PATH="${USER_FEATURES_PATH}"
export FEATURE_TESTS_PATH="${FEATURE_TESTS_PATH}"
export MODEL_PATH="${MODEL_PATH}"
export TEST_CATEGORY="${TEST_CATEGORY}"
export CI=true

echo "🔍 Проверка переменных для Docker Compose:"
echo "   USER_FEATURES_PATH: ${USER_FEATURES_PATH}"
echo "   FEATURE_TESTS_PATH: ${FEATURE_TESTS_PATH}"
echo "   MODEL_PATH: ${MODEL_PATH}"
echo ""

# Локальный запуск через tests/analyze/docker-compose.yml
docker compose -f docker-compose.yml build semantic-analyzer
docker compose -f docker-compose.yml run --rm \
    -e TEST_CATEGORY="${TEST_CATEGORY}" \
    -e CI=true \
    semantic-analyzer

echo ""
echo "✅ Демо завершено!"
echo "📊 Результаты анализа выведены выше"
echo ""
echo "💡 Для запуска на своих фичах используйте (не выходя из tests/analyze):"
echo "   TEST_CATEGORY=<категория_или_till-X> docker compose -f docker-compose.yml run --rm semantic-analyzer"
