import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100


class ResNet_KD(pl.LightningModule):
    def __init__(self, student_model:int, T, T1, T2, alpha, K):
        super(ResNet_KD, self).__init__()
        model_link = "chenyaofo/pytorch-cifar-models"
        self.teacher = torch.hub.load(model_link, "cifar100_resnet56", pretrained=True)
        self.teacher.eval()
        self.student = torch.hub.load(model_link, f"cifar100_resnet{student_model}", pretrained=False)
        self.T, self.T1, self.T2, self.alpha, self.k = T, T1, T2, alpha, K

        self.acc1 = Accuracy()
        self.acc5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.student(x)

    def configure_optimizers(self):
        return optim.Adam(self.student.parameters(), lr=0.001)

    def option1(self, student_logit, teacher_logit, T, T1, T2, K):
        new_teacher_logit = teacher_logit + torch.abs(torch.min(teacher_logit, dim=1).values.reshape(-1, 1))
        new_teacher_logit = new_teacher_logit / 2
        bar = torch.sort(new_teacher_logit, descending=True).values[:, K - 1] \
            .reshape(-1, 1).repeat(1, teacher_logit.shape[1])
        top = torch.where(bar <= new_teacher_logit, new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        bot = torch.where(bar > new_teacher_logit, new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        soft_label = F.softmax((top / T1) + (bot / T2), dim=1)
        soft_prediction = F.log_softmax(student_logit / T, dim=1)
        return F.kl_div(soft_prediction, soft_label)

    def FinalLoss_option1(self, teacher_logit, student_logit, labels, T, T1, T2, alpha, K):
        return (1. - alpha) * F.cross_entropy(student_logit, labels) \
               + (alpha * T1 * T2) * self.option1(student_logit, teacher_logit, T, T1, T2, K)

    # STEP
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        with torch.no_grad():
            self.teacher.eval()
            teacher_logit = self.teacher(inputs)
        student_logit = self.student(inputs)
        loss = self.FinalLoss_option1(teacher_logit, student_logit, labels, self.T, self.T1, self.T2, self.alpha, self.k)
        return {'loss': loss,
                'progress_bar':
                    {'ttop1_acc': self.acc1(teacher_logit, labels), 'ttop5_acc': self.acc5(teacher_logit, labels),
                     'stop1_acc': self.acc1(student_logit, labels), 'stop5_acc': self.acc5(student_logit, labels)}
                }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        teacher_logit = self.teacher(inputs)
        student_logit = self.student(inputs)
        loss = self.FinalLoss_option1(teacher_logit, student_logit, labels, self.T, self.T1, self.T2, self.alpha, self.k)
        return {'loss': loss,
                'progress_bar':
                    {'ttop1_acc': self.acc1(teacher_logit, labels), 'ttop5_acc': self.acc5(teacher_logit, labels),
                     'stop1_acc': self.acc1(student_logit, labels), 'stop5_acc': self.acc5(student_logit, labels)}
                }

    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)

    # EPOCH_END
    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_tacc1 = torch.stack([x['progress_bar']['ttop1_acc'] for x in outputs]).mean()
        train_tacc5 = torch.stack([x['progress_bar']['ttop5_acc'] for x in outputs]).mean()
        train_sacc1 = torch.stack([x['progress_bar']['stop1_acc'] for x in outputs]).mean()
        train_sacc5 = torch.stack([x['progress_bar']['stop5_acc'] for x in outputs]).mean()
        self.log_dict({'Train Loss': train_loss,
                       'Train TTop-1 Accuracy': train_tacc1 * 100,
                       'Train TTop-5 Accuracy': train_tacc5 * 100,
                       'Train Top-1 Accuracy': train_sacc1 * 100,
                       'Train Top-5 Accuracy': train_sacc5 * 100,
                       'step': self.current_epoch + 1.0})

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack([x['loss'] for x in outputs]).mean()
        valid_tacc1 = torch.stack([x['progress_bar']['ttop1_acc'] for x in outputs]).mean()
        valid_tacc5 = torch.stack([x['progress_bar']['ttop5_acc'] for x in outputs]).mean()
        valid_sacc1 = torch.stack([x['progress_bar']['stop1_acc'] for x in outputs]).mean()
        valid_sacc5 = torch.stack([x['progress_bar']['stop5_acc'] for x in outputs]).mean()
        self.log_dict({'Validation Loss': valid_loss,
                       'Validation TTop-1 Accuracy': valid_tacc1 * 100,
                       'Validation TTop-5 Accuracy': valid_tacc5 * 100,
                       'Validation Top-1 Accuracy': valid_sacc1 * 100,
                       'Validation Top-5 Accuracy': valid_sacc5 * 100,
                       'step': self.current_epoch + 1.0})

    # def test_epoch_end(self, outputs):
    #     test_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
    #     test_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
    #     self.log_dict({'Test Top-1 Accuracy': test_acc1 * 100,
    #                    'Test Top-5 Accuracy': test_acc5 * 100,
    #                    'step': 0.0})

    # DATASET, DATALOADER
    def prepare_data(self):
        CIFAR100(root='./data', train=True, download=True)
        CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop((32, 32), padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        self.dataset_train = CIFAR100(root='./data', train=True, transform=train_transform)
        self.dataset_valid = CIFAR100(root='./data', train=False, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=128, num_workers=4, pin_memory=True,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=128, num_workers=4, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset_test, batch_size=128, num_workers=4, pin_memory=True)


model = ResNet_KD(32, 2, 1.5, 2, 0.25, 5)
trainer = pl.Trainer(progress_bar_refresh_rate=10,
                     gpus=-1,
                     max_epochs=10,
                     )
# trainer.fit(model)
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100


class ResNet_Valid(pl.LightningModule):
    def __init__(self):
        super(ResNet_Valid, self).__init__()
        model_link = "chenyaofo/pytorch-cifar-models"
        self.teacher = torch.hub.load(model_link, "cifar100_resnet56", pretrained=True)

        self.acc1 = Accuracy()
        self.acc5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.teacher(x)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        teacher_logit = self.teacher(inputs)
        return {'progress_bar':
                    {'ttop1_acc': self.acc1(teacher_logit, labels), 'ttop5_acc': self.acc5(teacher_logit, labels)}
                }

        # def test_step(self, batch, batch_idx):
        #     return self.validation_step(batch, batch_idx)

        # EPOCH_END

    def test_epoch_end(self, outputs):
        train_tacc1 = torch.stack([x['progress_bar']['ttop1_acc'] for x in outputs]).mean()
        train_tacc5 = torch.stack([x['progress_bar']['ttop5_acc'] for x in outputs]).mean()
        print()
        print(train_tacc1, train_tacc5)


test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

dataset_valid = CIFAR100(root='./data', train=False, transform=test_transform)
test_dataloaders = DataLoader(dataset_valid, batch_size=128, num_workers=4, pin_memory=True)

model = ResNet_Valid()
trainer = pl.Trainer()
# trainer.fit(model)
trainer.test(model, test_dataloaders=test_dataloaders)