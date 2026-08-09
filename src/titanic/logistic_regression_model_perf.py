"""泰坦尼克号生存预测：带二次交叉特征的逻辑回归（性能增强版）。

相对基础版（logistic_regression_model.py）的主要差异：
- 在原始特征基础上构造两两乘积特征（含自身平方项）
- 用更多交互信息提升表达能力，例如 Age×Fare、Pclass×Sex_female

流程仍是：预处理 → 逻辑回归 → SGD 训练 → 验证集评估。
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LogisticRegressionModel(nn.Module):
    """逻辑回归模型：线性变换后接 Sigmoid，输出生存概率。

    虽然叫“逻辑回归”，输入可以是原始特征，也可以是交叉特征；
    模型本身仍是线性分类器，非线性来自特征工程而非网络深度。
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # 输入维数会更大（原始特征 + 交叉特征），但仍映射到 1 个 logit
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid 输出 (0, 1) 概率，便于用 BCE 与准确率阈值 0.5
        return torch.sigmoid(self.linear(x))


class TitanicDataset(Dataset):
    """带多项式交叉特征的泰坦尼克号数据集。

    特征构造策略：
    1. 清洗 + 独热编码（与基础版相同）
    2. 选取部分基特征，生成两两乘积项并标准化
    3. 再对基特征本身做 z-score 标准化

    mean / std 字典中既包含原始列，也包含交叉列（如 Age_Fare），
    统计量预先在完整训练集上算好并写死，保证训练/验证尺度一致。
    """

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = file_path

        # 原始特征 + 交叉特征的均值（交叉项命名规则：A_B 表示 A*B）
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
            "Pclass_Pclass": 5.704482,
            "Pclass_Age": 61.938151,
            "Pclass_SibSp": 1.198880,
            "Pclass_Parch": 0.983193,
            "Pclass_Fare": 53.052327,
            "Pclass_Sex_female": 0.754902,
            "Age_Age": 1092.761169,
            "Age_SibSp": 11.066415,
            "Age_Parch": 10.470476,
            "Age_Fare": 1104.142053,
            "Age_Sex_female": 10.204482,
            "SibSp_SibSp": 1.126050,
            "SibSp_Parch": 0.525210,
            "SibSp_Fare": 24.581262,
            "SibSp_Sex_female": 0.233894,
            "Parch_Parch": 0.913165,
            "Parch_Fare": 24.215465,
            "Parch_Sex_female": 0.259104,
            "Fare_Fare": 4000.200255,
            "Fare_Sex_female": 17.393698,
            "Sex_female_Sex_female": 0.365546,
        }

        # 对应标准差；分母接近 0 时标准化会不稳定，此处统计量均已预检可用
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
            "Pclass_Pclass": 3.447593,
            "Pclass_Age": 34.379609,
            "Pclass_SibSp": 2.603741,
            "Pclass_Parch": 2.236945,
            "Pclass_Fare": 52.407209,
            "Pclass_Sex_female": 1.118572,
            "Age_Age": 991.079188,
            "Age_SibSp": 19.093099,
            "Age_Parch": 29.164503,
            "Age_Fare": 1949.356185,
            "Age_Sex_female": 15.924481,
            "SibSp_SibSp": 3.428831,
            "SibSp_Parch": 1.561298,
            "SibSp_Fare": 70.185369,
            "SibSp_Sex_female": 0.639885,
            "Parch_Parch": 3.008314,
            "Parch_Fare": 77.207321,
            "Parch_Sex_female": 0.729143,
            "Fare_Fare": 19105.110593,
            "Fare_Sex_female": 43.568303,
            "Sex_female_Sex_female": 0.481921,
        }

        self.data = self._load_data()
        # 交叉特征会显著增加列数，feature_size 随之变大
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self) -> pd.DataFrame:
        """读取 CSV，构造交叉特征并标准化。"""
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

        # 参与交叉的基特征：含 Sex_female，可表达“舱位/年龄/票价 × 性别”等交互
        # 未把 Embarked_* 纳入交叉，以控制特征膨胀
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female"]

        # 双重循环 j 从 i 开始：生成上三角交叉，避免 A_B 与 B_A 重复
        # 当 i == j 时得到平方项（如 Age_Age = Age^2）
        for i in range(len(base_features)):
            for j in range(i, len(base_features)):
                left = base_features[i]
                right = base_features[j]
                cross_name = f"{left}_{right}"
                # 先用原始（未标准化）值做乘积，再对该乘积做 z-score
                # 这样交叉项的统计量与预先算好的 mean/std 字典一致
                df[cross_name] = (
                    df[left] * df[right] - self.mean[cross_name]
                ) / self.std[cross_name]

        # 基特征本身也做标准化（交叉项已用原始值算完，顺序不能颠倒）
        for feature_name in base_features:
            df[feature_name] = (df[feature_name] - self.mean[feature_name]) / self.std[
                feature_name
            ]
        return df

    def __len__(self) -> int:
        """样本总数。"""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (全部特征含交叉项, Survived 标签)。"""
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

DATA_DIR = Path(__file__).resolve().parents[2] / "dataset"
train_dataset = TitanicDataset(DATA_DIR / "train.csv")
validation_dataset = TitanicDataset(DATA_DIR / "validation.csv")

# 输入维 = 原始特征 + Embarked 独热 + 全部交叉项
model = LogisticRegressionModel(train_dataset.feature_size)
model.to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 100

# ---------------------------------------------------------------------------
# 训练：与基础版相同的 SGD + BCE 流程；差异主要来自更丰富的输入特征
# ---------------------------------------------------------------------------
for epoch in range(epochs):
    correct = 0
    step = 0
    total_loss = 0.0

    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features).squeeze()

        # 阈值 0.5 将概率转为类别，累计训练正确数
        correct += torch.sum((outputs >= 0.5) == labels)

        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / step:.4f}")
    print(f"Training Accuracy: {correct / len(train_dataset)}")

# ---------------------------------------------------------------------------
# 验证：不更新参数，只看交叉特征是否提升泛化准确率
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features).squeeze()
        correct += torch.sum((outputs >= 0.5) == labels)
    print(f"Validation Accuracy: {correct / len(validation_dataset)}")
