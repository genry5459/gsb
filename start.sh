#!/bin/bash
set -e

echo "========================================"
echo "Gold Micro Scalper — Railway Startup"
echo "========================================"

# Проверяем, что Railway передал переменные окружения
MISSING=0

if [ -z "$CAPITAL_API_KEY" ]; then
    echo "FATAL: CAPITAL_API_KEY is not set"
    echo "Go to Railway Dashboard -> your service -> Variables -> Add CAPITAL_API_KEY"
    MISSING=1
else
    echo "OK: CAPITAL_API_KEY is set (length: ${#CAPITAL_API_KEY})"
fi

if [ -z "$CAPITAL_LOGIN" ]; then
    echo "FATAL: CAPITAL_LOGIN is not set"
    MISSING=1
else
    echo "OK: CAPITAL_LOGIN is set (${CAPITAL_LOGIN:0:3}***)"
fi

if [ -z "$CAPITAL_PASSWORD" ]; then
    echo "FATAL: CAPITAL_PASSWORD is not set"
    MISSING=1
else
    echo "OK: CAPITAL_PASSWORD is set (length: ${#CAPITAL_PASSWORD})"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "========================================"
    echo "ENV CHECK FAILED"
    echo "========================================"
    echo "Sleeping 60s so you can read the logs..."
    sleep 60
    exit 1
fi

echo ""
echo "All required variables present. Starting bot..."
echo "========================================"

# Запускаем бота
exec python gold_micro_scalper_unified.py
