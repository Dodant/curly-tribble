from statistics import mean

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from xy_dataset import *

x_list, y_list = [], []
xy_pair = []
dataset = 2
filename = f'data/data-gp{dataset}.txt'
with open(filename, 'r') as f:
    for line in f:
        x, y = line.split(',')
        x_list.append(float(x))
        y_list.append(float(y))
        xy_pair.append((float(x), float(y)))


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.network(x)
        return x


net = Regressor()

best_fit = 100000
learning_rate = 0.00001
batch_size = 10
epochs = 1000
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

dataset = XY_Dataset(dataset)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    net.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        x = torch.unsqueeze(x, dim=1)
        output = net(x)
        loss = criterion(output, torch.unsqueeze(y, dim=1))
        loss.backward()
        optimizer.step()

    net.eval()
    fitness_score = mean([(net(torch.tensor(x, dtype=torch.float32).reshape(1)).item() - y) ** 2 for (x, y) in xy_pair]) ** 0.5
    print(f'EPO {epoch+1:>3d} - fitness: {fitness_score:.2f}')
    if best_fit > fitness_score: best_fit = fitness_score

print(f'best - {best_fit}')
plt.scatter(x_list, y_list)
plt.scatter(np.arange(-3, 3, 0.01),
            [net(torch.tensor(x, dtype=torch.float32).reshape(1)).detach().numpy()
             for x in np.arange(-3, 3, 0.01)])
plt.savefig(f'mlp_image/net_regressor_{dataset}_{best_fit:.2f}.png')
plt.close()
