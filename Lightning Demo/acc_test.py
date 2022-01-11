import argparse
import time
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pl_bolts.callbacks import PrintTableMetricsCallback
from torchmetrics import Accuracy

metric = Accuracy()

a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

c = [1, 1, 1, 1, 0]
d = [1, 1, 1, 1, 1]

a = torch.tensor(a)
b = torch.tensor(b)
c = torch.tensor(c)
d = torch.tensor(d)

print(a_acc := metric(a, b))
print(c_acc := metric(c, d))
print(f'mean: {((a_acc + c_acc)/2)*100:.2f}')
print(f'compute: {(metric.compute())*100:.2f}')
