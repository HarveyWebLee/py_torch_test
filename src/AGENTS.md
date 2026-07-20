# PyTorch 实验脚本

本目录存放 PyTorch 学习与性能实验脚本。

## 运行

```bash
uv run python src/tensor_use.py
uv run python src/tensor_tidu.py
```

## 约定

- 张量显式指定 `device`，支持 CPU 回退
- GPU 性能测试在计时前后调用 `torch.cuda.synchronize()`
- 新脚本命名：`tensor_<topic>.py` 或 `<topic>_benchmark.py`
