---
name: quality-gate
description: 运行项目的格式化、lint 与类型检查。在用户修改 Python 代码后、提交前、或询问代码质量时使用。
---

# 质量检查

## 快速检查（全部）

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

## 自动修复

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright
```

## 检查清单

- [ ] ruff format 无 diff
- [ ] ruff check 零告警
- [ ] pyright 零 error

任一失败时先修复再继续提交或 PR。
