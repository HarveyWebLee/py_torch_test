"""泰坦尼克号生存预测：基础逻辑回归示例。

流程概览：
1. 读取 train / validation CSV，做特征清洗与标准化
2. 用单层线性层 + Sigmoid 做二分类（逻辑回归）
3. 在训练集上用 SGD 优化二元交叉熵损失
4. 在验证集上评估准确率（不反向传播）
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LogisticRegressionModel(nn.Module):
    """逻辑回归模型：线性变换后接 Sigmoid，输出生存概率。

    数学形式：y_hat = sigmoid(W x + b)，其中 y_hat ∈ (0, 1)。
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # 将 input_dim 维特征映射为 1 个 logit（未归一化的得分）
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid 把 logit 压到 (0, 1)，可解释为“生还概率”
        return torch.sigmoid(self.linear(x))


class TitanicDataset(Dataset):
    """泰坦尼克号数据集：负责读 CSV、预处理，并按索引返回 (特征, 标签)。

    标准化使用的 mean / std 来自完整训练集统计，训练与验证集共用同一套数值，
    避免验证集信息泄漏到统计量中，也保证两侧特征尺度一致。
    """

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = file_path
        # 各数值特征在原始训练集上的均值（用于 z-score 标准化）
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910,
        }

        # 各数值特征在原始训练集上的标准差
        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274,
        }

        self.data = self._load_data()
        # 特征维数 = 全部列数 - 1（减去标签列 Survived）
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self) -> pd.DataFrame:
        """读取并预处理 CSV，返回可直接用于训练的 DataFrame。"""
        df = pd.read_csv(self.file_path)

        # 丢弃对预测帮助有限或难直接数值化的列
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

        # Age 缺失较多；此处直接删除缺失行（简单基线做法）
        df = df.dropna(subset=["Age"])

        # 类别特征独热编码：Sex -> Sex_female/Sex_male，Embarked -> C/Q/S
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

        # 对连续/有序数值特征做 z-score：x' = (x - mean) / std
        # 使不同量纲特征（如 Age 与 Fare）处于相近尺度，利于 SGD 收敛
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for feature_name in base_features:
            df[feature_name] = (df[feature_name] - self.mean[feature_name]) / self.std[
                feature_name
            ]
        return df

    def __len__(self) -> int:
        """样本总数，供 DataLoader 计算 epoch 长度。"""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按索引返回单条样本：(特征向量, 是否生还标签)。"""
        features = self.data.drop(columns=["Survived"]).iloc[index].values
        label = self.data["Survived"].iloc[index]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# 数据与模型准备
# ---------------------------------------------------------------------------

# 设备选择：CUDA → Apple MPS → CPU，保证无 GPU 时也能运行
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device} device")

# 项目根目录下的 dataset/（绝对路径，不依赖当前工作目录）
DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
train_dataset = TitanicDataset(DATA_DIR / "train.csv")
validation_dataset = TitanicDataset(DATA_DIR / "validation.csv")

# 输入维度由训练集特征列数决定
model = LogisticRegressionModel(train_dataset.feature_size)
model.to(device)  # 将参数与缓冲区搬到选定设备
model.train()  # 训练模式（对本模型影响很小，主要是习惯写法）

# 随机梯度下降；lr=0.1 对已标准化特征通常较合适
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 100

# ---------------------------------------------------------------------------
# 训练循环：每个 epoch 遍历全部训练 batch，更新权重并统计训练准确率
# ---------------------------------------------------------------------------
for epoch in range(epochs):
    correct = 0  # 本 epoch 预测正确的样本数
    step = 0  # batch 计数，用于平均 loss
    total_loss = 0.0

    # shuffle=True：每个 epoch 打乱样本顺序，减轻顺序偏差
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to(device)
        labels = labels.to(device)

        # 清空上一轮残留梯度，否则会累加到当前 batch
        optimizer.zero_grad()

        # squeeze：把形状从 [batch, 1] 压成 [batch]，与 labels 对齐
        outputs = model(features).squeeze()

        # 概率 >= 0.5 判为生还(1)，否则遇难(0)；统计本 batch 正确数
        correct += torch.sum((outputs >= 0.5) == labels)

        # 二元交叉熵：衡量预测概率分布与真实 0/1 标签的差距
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()

        loss.backward()  # 反向传播，计算各参数梯度
        optimizer.step()  # 按梯度更新参数

    # 平均 loss = 各 batch loss 之和 / batch 数
    print(f"Epoch {epoch + 1}, Loss: {total_loss / step:.4f}")
    # 训练准确率 = 本 epoch 训练集上猜对的比例
    print(f"Training Accuracy: {correct / len(train_dataset)}")

# ---------------------------------------------------------------------------
# 验证：关闭梯度，只前向推理，衡量泛化表现
# ---------------------------------------------------------------------------
model.eval()  # 评估模式
with torch.no_grad():  # 不建计算图，省显存、加快推理
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features).squeeze()
        correct += torch.sum((outputs >= 0.5) == labels)
    # 验证准确率：未见过的数据上的分类正确率（更反映泛化能力）
    print(f"Validation Accuracy: {correct / len(validation_dataset)}")
