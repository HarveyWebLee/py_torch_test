# py-torch-test

PyTorch 学习与实验项目，使用 uv 管理 Python 3.10.20 环境与 CUDA 12.9 版 PyTorch。

## 快速开始

```bash
uv sync --group dev
uv run python main.py
```

## 依赖安装

| 命令                    | 作用                                          |
| ----------------------- | --------------------------------------------- |
| `uv sync`               | 安装主依赖（torch、torchvision、tensorboard） |
| `uv sync --group dev`   | 额外安装开发工具（ruff、pyright、pre-commit） |
| `uv sync --group audio` | 额外安装 torchaudio                           |

Linux / Windows 上 `torch` / `torchvision` 从 CUDA 12.9 源安装；macOS 走 PyPI（含 MPS）。

## 开发命令

| 命令                           | 作用                                      |
| ------------------------------ | ----------------------------------------- |
| `uv run ruff format .`         | 格式化代码                                |
| `uv run ruff format --check .` | 仅检查格式，不改写文件                    |
| `uv run ruff check .`          | Lint 检查                                 |
| `uv run ruff check --fix .`    | 自动修复可修复的 lint                     |
| `uv run pyright`               | 类型检查                                  |
| `bash scripts/check.sh`        | 一键检查（format check + lint + pyright） |
| `bash scripts/fix.sh`          | 格式化 + 修复 + 类型检查                  |

等价一键检查：

```bash
uv run ruff format --check . && uv run ruff check . && uv run pyright
```

## 实验脚本

| 命令                                        | 作用                                      |
| ------------------------------------------- | ----------------------------------------- |
| `uv run python main.py`                     | 检查 PyTorch / CUDA 环境                  |
| `uv run python src/tensor_use.py`           | CPU vs GPU 矩阵乘法性能对比               |
| `uv run python src/tensor_tidu.py`          | 自动微分示例                              |
| `uv run python src/tensor_generate.py`      | 线性回归训练，写入 `runs/lr`              |
| `uv run python src/tensor_normalization.py` | 归一化对比训练，写入 `runs/normalization` |

## 查看 Loss 曲线（TensorBoard）

先运行带 `SummaryWriter` 的训练脚本，再启动 TensorBoard：

```bash
# 训练（写出事件文件）
uv run python src/tensor_generate.py
uv run python src/tensor_normalization.py

# 查看全部实验
uv run tensorboard --logdir=runs --host 127.0.0.1 --port 6806

# 或只看某一个实验
uv run tensorboard --logdir=runs/lr --host 127.0.0.1 --port 6806
uv run tensorboard --logdir=runs/normalization --host 127.0.0.1 --port 6806
```

浏览器打开 http://127.0.0.1:6806 ，在 Scalars 中查看 `loss/*` 曲线。

> **Windows 注意**：部分机器上默认端口 `6006` 落在系统保留端口段（如 Hyper-V），会报「以一种访问权限不允许的方式做了一个访问套接字的尝试」。请改用 `--port 6806`（或其它未被占用的端口）。  
> 「TensorFlow installation not found」仅为提示，不影响查看曲线。

## Git 钩子

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

提交信息格式：`feat(scope): 中文描述`

## Agent 指南

详见 [AGENTS.md](AGENTS.md)。
