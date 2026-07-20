---
name: pytorch-experiment
description: 在本项目中编写或调试 PyTorch 实验脚本。涉及张量、梯度、CUDA/GPU 性能测试时使用。
---

# PyTorch 实验

## 环境

```bash
uv sync
uv run python main.py          # 验证 CUDA
uv run python src/tensor_use.py
uv run python src/tensor_tidu.py
```

## 设备模板

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
tensor = torch.rand(3, 3, device=device)
```

## GPU / MPS 计时

```python
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
```

## 可选依赖

安装 torchaudio：`uv sync --group audio`
