from pathlib import Path

from titanic_dataset import TitanicDataset
from torch.utils.data import DataLoader

DATA_PATH = Path(__file__).resolve().parents[2] / "dataset" / "train.csv"
dataset = TitanicDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)
    break
