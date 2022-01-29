import torch
from torch.utils.data import Dataset

class CalibDataset(Dataset):
    def __init__(self, xs, ys, labs) -> None:
        super().__init__()
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32)
        self.labs = torch.tensor(labs, dtype=torch.float32)
    
    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.labs[idx]

