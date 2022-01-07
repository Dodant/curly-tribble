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
        return optim.SGD(self.parameters(), lr=1e-2)

    # STEP
    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        logits = self(x)
        loss = self.loss(logits, y)
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.acc1(logits, y), 'top5_acc': self.acc5(logits, y)}
                }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        logits = self(x)
        loss = self.loss(logits, y)
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.acc1(logits, y), 'top5_acc': self.acc5(logits, y)}
                }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # EPOCH_END
    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc1 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        train_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        self.log_dict({'Train Loss': train_loss,
                       'Train Top-1 Accuracy': train_acc1 * 100,
                       'Train Top-5 Accuracy': train_acc5 * 100,
                       'step': self.current_epoch + 1.0})

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack([x['loss'] for x in outputs]).mean()
        valid_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
        valid_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        self.log_dict({'Validation Loss': valid_loss,
                       'Validation Top-1 Accuracy': valid_acc1 * 100,
                       'Validation Top-5 Accuracy': valid_acc5 * 100,
                       'step': self.current_epoch + 1.0})

    def test_epoch_end(self, outputs):
        test_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
        test_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        self.log_dict({'Test Top-1 Accuracy': test_acc1 * 100,
                       'Test Top-5 Accuracy': test_acc5 * 100,
                       'step': 0.0})

    # DATASET, DATALOADER
    def prepare_data(self):
        datasets.MNIST('data', train=True, download=True)
        datasets.MNIST('data', train=False, download=True)

    def setup(self, stage=None):
        MNIST_train = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        self.dataset_test = datasets.MNIST('data', train=False, download=False, transform=transforms.ToTensor())
        self.dataset_train, self.dataset_valid = random_split(MNIST_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=128, num_workers=4, pin_memory=True,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=128, num_workers=4, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=128, num_workers=4, pin_memory=True)


model = MNIST_Classifier()
trainer = pl.Trainer(progress_bar_refresh_rate=10,
                     gpus=-1,
                     max_epochs=10,
                     precision=16
                     )
trainer.fit(model)
trainer.test(model)


# load model
# net = MNIST_Classifier.load_from_checkpoint(PATH)
# net.freeze()
# out = net(x)