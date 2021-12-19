import time
from statistics import mean
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from warmup_scheduler import GradualWarmupScheduler


from construct_ds import *
from dataset import *
from models import *
from plot_graph import *
from test_csv import *


EPOCHS = 500
TITLE = 'CNNwPTE'

with open('dataset/all.review.vec.txt', 'r') as f:
    dict_dict = dict()
    for line in f:
        values = line.split()
        word = values[0]
        dict_dict[word] = np.array(values[1:]).astype(float)
    f.close()

review_dataset = Embed_ReviewDataset(dict_dict)
trainset, vaildset = train_test_split(review_dataset, test_size=0.1)
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, drop_last=True)  # 1600
valid_loader = DataLoader(vaildset, batch_size=16, shuffle=False, drop_last=True)  # 400

reviewer = Embed_CNN_review(100).cuda()


optimizer = optim.RMSprop(reviewer.parameters(), lr=0.0001)
criterion = nn.MSELoss().cuda()
# criterion = nn.BCELoss()

train_loss_history, train_acc_history = [], []
valid_loss_history, valid_acc_history = [], []

st = time.time()
best_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc, train_sample, train_correct = 0, 0, 0, 0
    valid_loss, valid_acc, valid_sample, valid_correct = 0, 0, 0, 0

    reviewer.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = reviewer(x)
        loss = criterion(y, out.squeeze())
        loss.backward()
        optimizer.step()

        prediction = torch.round(out)
        train_correct += torch.sum((prediction == y).int())
        train_loss += loss.item() * len(x)
        train_sample += len(x)

    else:
        reviewer.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                out = reviewer(x)
                loss = criterion(y, out.squeeze())

                prediction = torch.round(out)
                valid_correct += torch.sum((prediction == y).int())
                valid_loss += loss.item() * len(x)
                valid_sample += len(x)

    epoch_loss = train_loss / train_sample
    epoch_acc = train_correct * 100 / train_sample
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)

    valid_epoch_loss = valid_loss / valid_sample
    valid_epoch_acc = valid_correct * 100 / valid_sample
    valid_loss_history.append(valid_epoch_loss)
    valid_acc_history.append(valid_epoch_acc)

    print(f'EPO {epoch+1:>3d} | tl - {epoch_loss:.5f}, ta - {epoch_acc:.2f}% | '
          f'vl - {valid_epoch_loss:.5f}, va - {valid_epoch_acc:.2f}%')

    if valid_epoch_acc > best_acc:
        embed_make_submission(reviewer, f"{TITLE}-{epoch+1}-va{valid_epoch_acc}", dict_dict)
        best_acc = valid_epoch_acc

print(f'Training Time - {time.time() - st:.2f}s')


# training/testing objective over time
draw_graph(train_loss_history, valid_loss_history, TITLE, "Loss")
# training/testing accuracy over time
draw_graph(train_acc_history, valid_acc_history, TITLE, "Acc")

# make submission
embed_make_submission(reviewer, TITLE, dict_dict)
