#!/bin/bash
set -e

echo "========================================"
echo "Gold Micro Scalper — OANDA Startup"
echo "========================================"

MISSING=0

if [ -z "$OANDA_API_TOKEN" ]; then
    echo "FATAL: OANDA_API_TOKEN is not set"
    echo "Go to Railway Dashboard -> your service -> Variables -> Add OANDA_API_TOKEN"
    MISSING=1
else
    echo "OK: OANDA_API_TOKEN is set (length: ${#OANDA_API_TOKEN})"
fi

if [ -z "$OANDA_ACCOUNT_ID" ]; then
    echo "FATAL: OANDA_ACCOUNT_ID is not set"
    echo "Go to Railway Dashboard -> your service -> Variables -> Add OANDA_ACCOUNT_ID"
    MISSING=1
else
    echo "OK: OANDA_ACCOUNT_ID is set ($OANDA_ACCOUNT_ID)"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Get your token at: https://ft.oanda.com -> Manage API Access"
    echo "Account ID is in the top-left corner of the OANDA dashboard"
    echo "========================================"
    sleep 60
    exit 1
fi

echo ""
echo "All required variables present. Starting bot..."
echo "========================================"

exec python gold_micro_scalper_unified.py
