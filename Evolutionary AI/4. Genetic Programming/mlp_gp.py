from statistics import mean

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

dataset = 1
xy_pair = []
x_list, y_list = [], []
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
            nn.Linear(200, 160, bias=False),
            nn.ReLU(),
            nn.Linear(160, 320, bias=False),
            nn.ReLU(),
            nn.Linear(320, 160, bias=False),
            nn.ReLU(),
            nn.Linear(160, 200, bias=False)
        )

    def forward(self, x):
        x = self.network(x)
        return x


net = Regressor()

learning_rate = 0.001
weight_decay = 0.1
batch_size = 16
epochs = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

X = torch.Tensor(x_list)
Y = torch.Tensor(y_list)


for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    print(f'{loss.item()}   ---   fitness : {mean([net(X)]):.1f}')

plt.scatter(x_list, y_list)
plt.scatter(np.arange(-3, 3, 0.01), [net(x) for x in np.arange(-3, 3, 0.01)])
plt.savefig(f'net_regressor.png')
plt.close()

fitness_score = mean([(net(x) - y) ** 2 for (x, y) in xy_pair])
