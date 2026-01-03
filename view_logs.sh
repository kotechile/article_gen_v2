#!/bin/bash
# Quick log viewer script for Content Generator V2

LOG_DIR="/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/logs"

echo "=== Flask Server Logs (last 20 lines) ==="
tail -20 "$LOG_DIR/flask.log" 2>/dev/null || echo "No Flask logs found"

echo ""
echo "=== Celery Worker Logs (last 20 lines) ==="
tail -20 "$LOG_DIR/celery.log" 2>/dev/null || echo "No Celery logs found"

echo ""
echo "=== Active Processes ==="
ps aux | grep -E "python.*app\.py|celery.*worker" | grep -v grep


