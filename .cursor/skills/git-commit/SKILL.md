---
name: git-commit
description: 按项目规范创建 git 提交。用户要求提交、写 commit message 或准备 PR 时使用。
---

# Git 提交工作流

## 前置检查

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

## 提交步骤

1. `git status` 与 `git diff` 查看变更
2. 仅暂存相关文件：`git add <files>`
3. 使用 HEREDOC 提交（subject 中文）：

```bash
git commit -m "$(cat <<'EOF'
feat(scope): 简短描述

EOF
)"
```

## 安全规则

- 不 force push main/master
- 不跳过 hooks（除非用户明确要求）
- 不提交 `.venv`、密钥、大文件

## type 选择

| 变更类型    | type     |
| ----------- | -------- |
| 新功能/脚本 | feat     |
| 修复 bug    | fix      |
| 文档        | docs     |
| 仅格式      | style    |
| 重构        | refactor |
| 依赖/构建   | build    |
| 杂项        | chore    |
