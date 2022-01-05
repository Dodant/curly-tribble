import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class MNIST_Classifier(pl.LightningModule):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)

        self.loss = nn.CrossEntropyLoss()
        self.acc1 = Accuracy()
        self.acc5 = Accuracy(top_k=5)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        do = self.dropout(h1+h2)
        logits = self.l3(do)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        logits = self(x)
        loss = self.loss(logits, y)
        acc_1 = self.acc1(logits, y)
        acc_5 = self.acc5(logits, y)
        pbar = {'train_top1': acc_1, 'train_top5': acc_5}
        self.log("train_loss", loss)
        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_top1'] = results['progress_bar']['train_top1']
        results['progress_bar']['val_top5'] = results['progress_bar']['train_top5']
        del results['progress_bar']['train_top1']
        del results['progress_bar']['train_top5']
        return results

    def test_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['test_top1'] = results['progress_bar']['train_top1']
        results['progress_bar']['test_top5'] = results['progress_bar']['train_top5']
        del results['progress_bar']['train_top1']
        del results['progress_bar']['train_top5']
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc1 = torch.tensor([x['progress_bar']['val_top1'] for x in val_step_outputs]).mean()
        avg_val_acc5 = torch.tensor([x['progress_bar']['val_top5'] for x in val_step_outputs]).mean()
        pbar = {'avg_val_acc1': avg_val_acc1, 'avg_val_acc5': avg_val_acc5}
        self.log('validation_loss', avg_val_loss)
        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def prepare_data(self):
        datasets.MNIST('data', train=True, download=True)
        datasets.MNIST('data', train=False, download=True)

    def setup(self, stage=None):
        MNIST_train = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        self.dataset['test'] = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
        self.dataset['train'], self.dataset['valid'] = random_split(MNIST_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=32, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset['valid'], batch_size=32, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=32, num_workers=4, pin_memory=True)

model = MNIST_Classifier()
trainer = pl.Trainer(progress_bar_refresh_rate=10, gpus=-1, max_epochs=5)
trainer.fit(model)
# trainer.test(model)