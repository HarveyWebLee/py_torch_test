#!/usr/bin/env bash
# 编辑 Python 文件后自动运行 ruff format
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('file_path', ''))
" 2>/dev/null || echo "")

if [[ -z "$file_path" ]]; then
  exit 0
fi

if [[ ! "$file_path" =~ \.py$ ]]; then
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  exit 0
fi

uv run ruff format "$file_path" 2>/dev/null || true
exit 0
