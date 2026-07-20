#!/usr/bin/env bash
# 格式化并修复可自动修复的 lint 问题
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format"
uv run ruff format .

echo "==> ruff check --fix"
uv run ruff check --fix .

echo "==> pyright"
uv run pyright

echo "✓ 完成"
