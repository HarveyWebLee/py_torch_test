"""送餐时间线性回归：对比无归一化 / 最大值归一化 / 标准化（见 rethink.fun 6.8）。"""

from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_delivery_data(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # time = 2 * lights + 0.01 * distance + 5（无噪声）
    inputs = torch.tensor(
        [[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [[19], [31], [14], [15], [43]],
        dtype=torch.float32,
        device=device,
    )
    return inputs, labels


def train_linear(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epoch: int,
    lr: float,
    writer: SummaryWriter,
    tag: str,
    print_grad_first_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = inputs.device
    w = torch.ones(2, 1, requires_grad=True, device=device)
    b = torch.ones(1, requires_grad=True, device=device)

    for i in range(epoch):
        outputs = inputs @ w + b
        loss = torch.mean(torch.square(outputs - labels))
        writer.add_scalar(f"loss/{tag}", loss.item(), i)

        loss.backward()
        w_grad = w.grad
        b_grad = b.grad
        assert w_grad is not None and b_grad is not None

        if print_grad_first_step and i == 0:
            print(f"[{tag}] step0 loss={loss.item():.6f}")
            print(f"[{tag}] step0 w.grad={w_grad.tolist()}")

        with torch.no_grad():
            w -= w_grad * lr
            b -= b_grad * lr

        w_grad.zero_()
        b_grad.zero_()

        if i == 0 or i == epoch - 1 or (i + 1) % max(epoch // 5, 1) == 0:
            print(f"[{tag}] epoch={i + 1}/{epoch} loss={loss.item():.6f}")

    return w.detach(), b.detach()


def main() -> None:
    device = select_device()
    print(f"Using device: {device}")

    inputs_raw, labels = load_delivery_data(device)
    log_dir = Path(__file__).resolve().parents[1] / "runs" / "normalization"
    writer = SummaryWriter(log_dir=str(log_dir))

    print("\n=== 1) 无归一化（特征量纲差异大，学习率必须极小）===")
    w1, b1 = train_linear(
        inputs_raw,
        labels,
        epoch=200,
        lr=1e-7,
        writer=writer,
        tag="raw",
        print_grad_first_step=True,
    )
    print(f"训练后 w={w1.tolist()}, b={b1.tolist()}")

    print("\n=== 2) 最大值归一化（feature / max → [0,1]）===")
    # lights 最大 4，distance 最大 3000
    inputs_max = inputs_raw / torch.tensor([4.0, 3000.0], device=device)
    w2, b2 = train_linear(
        inputs_max,
        labels,
        epoch=1000,
        lr=0.5,
        writer=writer,
        tag="max_norm",
    )
    print(f"训练后 w={w2.tolist()}, b={b2.tolist()}")

    print("\n=== 3) 标准化（(x - mean) / std）+ 预测 ===")
    mean = inputs_raw.mean(dim=0)
    std = inputs_raw.std(dim=0)
    inputs_std = (inputs_raw - mean) / std
    w3, b3 = train_linear(
        inputs_std,
        labels,
        epoch=2000,
        lr=0.1,
        writer=writer,
        tag="standardize",
    )
    print(f"mean={mean.tolist()}, std={std.tolist()}")
    print(f"训练后 w={w3.tolist()}, b={b3.tolist()}")

    # 预测时必须用训练时的 mean/std
    new_input = torch.tensor([[3.0, 2500.0]], device=device)
    new_input_norm = (new_input - mean) / std
    predict = new_input_norm @ w3 + b3
    # 真值：2*3 + 0.01*2500 + 5 = 36
    print(f"新样本 [lights=3, distance=2500] 预测={predict.item():.4f}（真值约 36）")

    writer.close()
    print(f"\nTensorBoard 日志: {log_dir}")
    print(f"查看曲线: uv run tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
