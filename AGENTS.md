# AGENTS.md

本文件为 AI Agent 提供项目级操作指南。与 `.cursor/rules/` 互补：此处侧重命令、验证流程与边界；规则文件侧重编码细节。

## 项目概览

| 项     | 值                                 |
| ------ | ---------------------------------- |
| 语言   | Python 3.10.20                     |
| 框架   | PyTorch 2.5.1 + torchvision 0.20.1 |
| CUDA   | 12.1（可选，需 NVIDIA GPU）        |
| 包管理 | [uv](https://docs.astral.sh/uv/)   |
| 源码   | `main.py`、`src/`                  |

## 常用命令

```bash
# 安装依赖
uv sync                 # 主依赖：torch / torchvision / tensorboard
uv sync --group dev     # 开发工具：ruff / pyright / pre-commit
uv sync --group audio   # 可选：torchaudio

# 运行入口
uv run python main.py

# 实验脚本
uv run python src/tensor_use.py            # CPU vs GPU 矩阵乘法
uv run python src/tensor_tidu.py           # 自动微分
uv run python src/tensor_generate.py       # 线性回归，日志 → runs/lr
uv run python src/tensor_normalization.py  # 归一化对比，日志 → runs/normalization

# 查看 Loss 曲线（Windows 避免默认 6006：该端口常被系统保留）
uv run tensorboard --logdir=runs --host 127.0.0.1 --port 6806

# 格式化 / Lint / 类型检查
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .
uv run pyright

# 一键质量检查
uv run ruff format --check . && uv run ruff check . && uv run pyright
```

更多说明见 [README.md](README.md)。

## 开发工作流

1. 在 `src/` 新增或修改实验脚本
2. 修改后运行格式化与检查（见上）
3. 本地验证：`uv run python <script>.py`
4. 仅在用户要求时提交；提交前确保质量检查通过

## 代码规范

- **类型**：公开函数必须有类型注解；`pyright` mode 为 `basic`
- **格式**：ruff format，行宽 88，双引号
- **导入**：isort 由 ruff 管理，first-party 为 `src`
- **PyTorch**：设备选择必须有 CPU 回退；GPU 计时时调用 `torch.cuda.synchronize()`
- **语言**：注释与 Agent 回复默认简体中文

## 目录结构

```
py_torch_test/
├── main.py                     # 入口：检查 PyTorch / CUDA
├── src/
│   ├── tensor_use.py           # GPU vs CPU 矩阵乘法
│   ├── tensor_tidu.py          # 自动微分示例
│   ├── tensor_generate.py      # 线性回归 + TensorBoard（runs/lr）
│   └── tensor_normalization.py # 归一化对比 + TensorBoard（runs/normalization）
├── runs/                       # TensorBoard 事件文件（本地生成，勿提交）
├── .cursor/
│   ├── rules/                  # Cursor 规则（.mdc）
│   ├── skills/                 # Agent 技能
│   ├── hooks/                  # Cursor 钩子脚本
│   ├── hooks.json              # 钩子配置
│   └── mcp.json                # MCP 服务器配置
├── AGENTS.md                   # 本文件
└── pyproject.toml              # 依赖与工具配置
```

## Git 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)，subject 使用中文：

```
<type>(<scope>): <subject>
```

| type     | 用途      |
| -------- | --------- |
| feat     | 新功能    |
| fix      | 修复      |
| docs     | 文档      |
| style    | 仅格式    |
| refactor | 重构      |
| perf     | 性能      |
| test     | 测试      |
| build    | 构建/依赖 |
| ci       | CI        |
| chore    | 杂项      |

**示例**：`feat(tensor): 添加梯度计算示例`

安装 pre-commit 钩子：

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## 禁止触碰

- `.venv/`、`uv.lock`（非依赖变更时）
- 模型权重、数据集、密钥文件
- 未经用户同意的 `git push --force`、`git reset --hard`

## Cursor 集成

| 资源   | 路径                        | 说明                              |
| ------ | --------------------------- | --------------------------------- |
| Rules  | `.cursor/rules/*.mdc`       | 自动注入的编码规则                |
| Skills | `.cursor/skills/*/SKILL.md` | 按需加载的工作流                  |
| Hooks  | `.cursor/hooks.json`        | 编辑后自动 format、危险 git 拦截  |
| MCP    | `.cursor/mcp.json`          | MCP 服务器（见 `.cursor/MCP.md`） |

### 推荐 Skills

- `quality-gate` — 运行 format / lint / typecheck
- `pytorch-experiment` — PyTorch 实验模板
- `git-commit` — 规范提交流程

## 验证清单

完成代码修改后确认：

- [ ] `uv run ruff format --check .` 通过
- [ ] `uv run ruff check .` 零告警
- [ ] `uv run pyright` 零 error
- [ ] 相关脚本可运行且无运行时错误
