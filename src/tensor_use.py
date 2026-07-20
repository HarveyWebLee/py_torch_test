import time

import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


device = select_device()
print(f"Using device: {device}")

size = 10000
A_cpu = torch.rand(size, size)
B_cpu = torch.rand(size, size)

start_cpu = time.time()
_ = torch.mm(A_cpu, B_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU time: {cpu_time:.6f} sec")

if device.type == "cpu":
    print("Accelerator not available, skipping GPU/MPS test.")
else:
    A_acc = A_cpu.to(device)
    B_acc = B_cpu.to(device)
    # 预热，避免首次 kernel 编译影响计时
    _ = torch.mm(A_acc, B_acc)
    synchronize(device)

    start_acc = time.time()
    _ = torch.mm(A_acc, B_acc)
    synchronize(device)
    end_acc = time.time()
    acc_time = end_acc - start_acc

    print(f"{device.type.upper()} time: {acc_time:.6f} sec")
