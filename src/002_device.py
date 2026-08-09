import torch  # 导入 PyTorch 核心库
from torch import nn  # 导入神经网络模块

# 按可用性选择设备：优先 CUDA，其次 Apple MPS，最后 CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")  # 打印当前使用的计算设备


class NeuralNetwork(nn.Module):  # 定义全连接分类网络
    def __init__(self):
        super().__init__()  # 初始化父类 nn.Module
        self.flatten = nn.Flatten()  # 将输入展平为 [batch, features]
        self.linear_relu_stack = nn.Sequential(  # 线性层与 ReLU 堆叠
            nn.Linear(28 * 28, 512),  # 784 → 512 全连接
            nn.ReLU(),  # 非线性激活
            nn.Linear(512, 512),  # 512 → 512 全连接
            nn.ReLU(),  # 非线性激活
            nn.Linear(512, 10),  # 512 → 10 类别 logits
        )

    def forward(self, x):  # 前向传播
        x = self.flatten(x)  # 展平图像
        return self.linear_relu_stack(x)  # 返回未归一化的类别 logits


model = NeuralNetwork().to(device)  # 创建模型并移到选定设备
print("---- model ----\n", model)  # 打印网络结构

X = torch.rand(1, 28, 28, device=device)  # 随机生成一张 28×28 假输入
logits = model(X)  # 前向推理得到 logits
pred_probable = nn.Softmax(dim=1)(logits)  # 沿类别维做 Softmax 得概率
y_pred = pred_probable.argmax(1)  # 取概率最大的类别索引
print(f"---- Predicted class: {y_pred} ----")  # 打印预测类别
