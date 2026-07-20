#!/usr/bin/env bash
# 拦截危险的 git 命令，提示用户确认
set -euo pipefail

input=$(cat)
command=$(echo "$input" | python -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('command', ''))
" 2>/dev/null || echo "")

if [[ -z "$command" ]]; then
  echo '{ "permission": "allow" }'
  exit 0
fi

if echo "$command" | grep -qE 'git push --force|git reset --hard|git clean -f'; then
  cat <<'EOF'
{
  "permission": "ask",
  "user_message": "检测到可能破坏历史的 git 命令，请确认是否继续。",
  "agent_message": "Hook 拦截了危险的 git 操作，需用户明确批准。"
}
EOF
  exit 0
fi

echo '{ "permission": "allow" }'
exit 0
