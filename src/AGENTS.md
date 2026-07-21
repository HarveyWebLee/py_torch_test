# PyTorch 实验脚本

本目录存放 PyTorch 学习与性能实验脚本。

## 运行

| 命令                                        | 作用                                      |
| ------------------------------------------- | ----------------------------------------- |
| `uv run python src/tensor_use.py`           | CPU vs GPU 矩阵乘法性能对比               |
| `uv run python src/tensor_tidu.py`          | 自动微分示例                              |
| `uv run python src/tensor_generate.py`      | 线性回归训练，写入 `runs/lr`              |
| `uv run python src/tensor_normalization.py` | 归一化对比训练，写入 `runs/normalization` |

## 查看 Loss 曲线

```bash
uv run tensorboard --logdir=runs --host 127.0.0.1 --port 6806
```

浏览器打开 http://127.0.0.1:6806 。Windows 上勿用默认 `6006`（常被系统保留端口段占用）。

## 约定

- 张量显式指定 `device`，支持 CPU 回退
- GPU 性能测试在计时前后调用 `torch.cuda.synchronize()`
- 新脚本命名：`tensor_<topic>.py` 或 `<topic>_benchmark.py`
