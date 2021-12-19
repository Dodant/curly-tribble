import torch
from torch.utils.data import Dataset


class XY_Dataset(Dataset):
    def __init__(self, dataset):
        self.x_list, self.y_list = [], []

        filename = f'data/data-gp{dataset}.txt'
        with open(filename, 'r') as f:
            for line in f:
                x, y = line.split(',')
                self.x_list.append(torch.tensor(float(x), dtype=torch.float32))
                self.y_list.append(torch.tensor(float(y), dtype=torch.float32))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]
