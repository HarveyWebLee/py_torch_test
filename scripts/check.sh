#!/usr/bin/env bash
# 一键运行格式化、lint 与类型检查
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> ruff check"
uv run ruff check .

echo "==> pyright"
uv run pyright

echo "✓ 全部检查通过"
