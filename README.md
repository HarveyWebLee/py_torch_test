# py-torch-test

PyTorch 学习与实验项目，使用 uv 管理 Python 3.10.20 环境与 CUDA 12.1 版 PyTorch。

## 快速开始

```bash
uv sync --group dev
uv run python main.py
```

## 开发命令

| 命令                        | 说明                     |
| --------------------------- | ------------------------ |
| `uv run ruff format .`      | 格式化代码               |
| `uv run ruff check .`       | Lint 检查                |
| `uv run ruff check --fix .` | 自动修复 lint            |
| `uv run pyright`            | 类型检查                 |
| `bash scripts/check.sh`     | 一键检查                 |
| `bash scripts/fix.sh`       | 格式化 + 修复 + 类型检查 |

## 实验脚本

```bash
uv run python src/tensor_use.py   # CPU vs GPU 矩阵乘法
uv run python src/tensor_tidu.py  # 自动微分
```

## Git 钩子

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

提交信息格式：`feat(scope): 中文描述`

## Agent 指南

详见 [AGENTS.md](AGENTS.md)。
