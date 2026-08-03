from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)  # 打印时显示所有列

# 项目根目录下的 dataset/train.csv（绝对路径）
DATA_PATH = Path(__file__).resolve().parents[2] / "dataset" / "train.csv"
df = pd.read_csv(DATA_PATH)

# 去除不需要的列
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# 去除 Age 缺失的样本
df = df.dropna(subset=["Age"])

# 对 Sex 和 Embarked 做独热编码
df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

print(df.head(10))