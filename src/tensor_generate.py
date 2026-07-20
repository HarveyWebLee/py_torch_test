from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter

# 确保 CUDA / MPS 可用，否则回退 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 生成数据
inputs = torch.rand(
    100, 3
)  # 随机生成shape为(100,3)的tensor，里边每个元素的值都是0-1之间
weights = torch.tensor([[1.1], [2.2], [3.3]])  # 预设的权重
bias = torch.tensor(4.4)  # 预设的bias
targets = (
    inputs @ weights + bias + 0.1 * torch.randn(100, 1)
)  # 增加一些误差，模拟真实情况

log_dir = Path(__file__).resolve().parents[1] / "runs" / "lr"
writer = SummaryWriter(log_dir=str(log_dir))

# 初始化参数时直接放在加速设备上，并启用梯度追踪
w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1,), requires_grad=True, device=device)

# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

# 设置超参数
epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print("loss:", loss.item())
    # 记录 loss：标签、值、步数
    writer.add_scalar("loss/train", loss.item(), i)

    loss.backward()
    w_grad = w.grad
    b_grad = b.grad
    assert w_grad is not None and b_grad is not None

    with torch.no_grad():  # 下边的计算不需要跟踪梯度
        w -= lr * w_grad
        b -= lr * b_grad

    # 清零梯度
    w_grad.zero_()
    b_grad.zero_()

writer.close()

print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)
print(f"TensorBoard 日志目录: {log_dir}")
print(f"查看曲线: uv run tensorboard --logdir={log_dir}")
