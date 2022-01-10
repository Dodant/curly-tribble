import argparse
import time
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torchmetrics import Accuracy

warnings.filterwarnings(action='ignore')
pl.seed_everything(42)

class ResNet_KD(pl.LightningModule):
    def __init__(self, student_model: int, T, T1, T2, alpha, K, option, batch):
        super(ResNet_KD, self).__init__()
        self.save_hyperparameters()
        model_link = 'chenyaofo/pytorch-cifar-models'
        self.teacher = torch.hub.load(model_link, 'cifar100_resnet56', pretrained=True)
        self.student = torch.hub.load(model_link, f'cifar100_resnet{student_model}', pretrained=False)

        self.acc1, self.acc5 = Accuracy(), Accuracy(top_k=5)
        self.best_acc1, self.best_acc5, self.best_epoch = 0.0, 0.0, 0

        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.configure_optimizers(), T_max=400)

    def forward(self, x):
        return self.student(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    def option0(self, student_logit, teacher_logit, labels):
        new_teacher_logit = teacher_logit + torch.abs(torch.min(teacher_logit, dim=1).values.reshape(-1, 1))
        new_teacher_logit /= 2
        soft_label = F.softmax(new_teacher_logit / self.hparams.T, dim=1)
        soft_prediction = F.log_softmax(student_logit / self.hparams.T, dim=1)
        return F.kl_div(soft_prediction, soft_label)

    def option1(self, student_logit, teacher_logit, labels):
        new_teacher_logit = teacher_logit + torch.abs(torch.min(teacher_logit, dim=1).values.reshape(-1, 1))
        new_teacher_logit /= 2
        bar = torch.sort(new_teacher_logit, descending=True).values[:, self.hparams.K - 1] \
            .reshape(-1, 1).repeat(1, teacher_logit.shape[1])
        top = torch.where(bar <= new_teacher_logit, new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        bot = torch.where(bar > new_teacher_logit, new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        soft_label = F.softmax((top / self.hparams.T1) + (bot / self.hparams.T2), dim=1)
        soft_prediction = F.log_softmax(student_logit / self.hparams.T, dim=1)
        return F.kl_div(soft_prediction, soft_label)

    def option2(self, student_logit, teacher_logit, labels):
        new_teacher_logit = teacher_logit + torch.abs(torch.min(teacher_logit, dim=1).values.reshape(-1, 1))
        new_teacher_logit /= 2
        bar = torch.sort(new_teacher_logit, descending=True).values[:, self.hparams.K - 1] \
            .reshape(-1, 1).repeat(1, teacher_logit.shape[1])
        without_gt = new_teacher_logit.scatter(1, labels.view(-1, 1), 0)
        top = torch.where(bar <= without_gt, without_gt, torch.zeros(1, device=torch.device('cuda')))
        bot = torch.where(bar > without_gt, without_gt, torch.zeros(1, device=torch.device('cuda')))
        gt = torch.where(torch.zeros_like(new_teacher_logit).scatter(1, labels.view(-1, 1), 1) == 1.,
                         new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        top, bot = top / self.hparams.T1, bot / self.hparams.T2
        soft_label = F.softmax(top + bot + gt, dim=1)
        soft_prediction = F.log_softmax(student_logit / self.hparams.T, dim=1)
        return F.kl_div(soft_prediction, soft_label)

    def option3(self, student_logit, teacher_logit, labels):
        new_teacher_logit = teacher_logit + torch.abs(torch.min(teacher_logit, dim=1).values.reshape(-1, 1))
        new_teacher_logit /= 2
        bar = torch.sort(new_teacher_logit, descending=True).values[:, self.hparams.K - 1] \
            .reshape(-1, 1).repeat(1, teacher_logit.shape[1])
        new_teacher_logit = torch.where(bar <= new_teacher_logit,
                                        new_teacher_logit, torch.zeros(1, device=torch.device('cuda')))
        soft_label = F.softmax(new_teacher_logit / self.hparams.T, dim=1)
        soft_prediction = F.log_softmax(student_logit / self.hparams.T, dim=1)
        return F.kl_div(soft_prediction, soft_label)

    def FinalLoss(self, student_logit, teacher_logit, labels):
        if self.hparams.option == 3: self.hparams.T1, self.hparams.T2 = self.hparams.T, self.hparams.T
        return (1. - self.hparams.alpha) * \
               F.cross_entropy(student_logit, labels) + \
               (self.hparams.alpha * self.hparams.T1 * self.hparams.T2) * \
               eval(f'self.option{self.hparams.option}(student_logit, teacher_logit, labels)')

    # STEP
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.teacher.eval()
        with torch.no_grad():
            teacher_logit = self.teacher(inputs)
        student_logit = self.student(inputs)
        loss = self.FinalLoss(student_logit, teacher_logit, labels)
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            self.sched.step()
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.acc1(student_logit, labels), 'top5_acc': self.acc5(student_logit, labels)}
                }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        student_logit = self.student(inputs)
        teacher_logit = self.teacher(inputs)
        loss = self.FinalLoss(student_logit, teacher_logit, labels)
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.acc1(student_logit, labels), 'top5_acc': self.acc5(student_logit, labels)}
                }

    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)

    # EPOCH_END
    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
        train_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        self.log_dict({'Train Loss': train_loss,
                       'Train Top-1 Accuracy': train_acc1 * 100,
                       'Train Top-5 Accuracy': train_acc5 * 100,
                       'step': self.current_epoch + 1})

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack([x['loss'] for x in outputs]).mean()
        valid_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
        valid_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
        self.log_dict({'Validation Loss': valid_loss,
                       'Validation Top-1 Accuracy': valid_acc1 * 100,
                       'Validation Top-5 Accuracy': valid_acc5 * 100,
                       'step': self.current_epoch + 1})
        if self.best_acc1 < valid_acc1:
            self.best_acc1, self.best_acc5, self.best_epoch = valid_acc1, valid_acc5, self.current_epoch + 1

    # def test_epoch_end(self, outputs):
    #     test_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
    #     test_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
    #     self.log_dict({'Test Top-1 Accuracy': test_acc1 * 100,
    #                    'Test Top-5 Accuracy': test_acc5 * 100,
    #                    'step': 0.0})

    # DATASET, DATALOADER
    def prepare_data(self):
        datasets.CIFAR100(root='./data', train=True, download=True)
        datasets.CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop((32, 32), padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

        self.dataset_train = datasets.CIFAR100(root='./data', train=True, transform=train_transform)
        self.dataset_valid = datasets.CIFAR100(root='./data', train=False, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch, num_workers=4, pin_memory=True,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.hparams.batch, num_workers=4, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset_test, batch_size=128, num_workers=4, pin_memory=True)


def kd_train_and_test(student_model, T, T1, T2, alpha, K, option, batch, epochs):
    print('============================================================\n'
          'CONFIG\n'
          f'student_model = {student_model} | option = {option} | alpha = {alpha}')
    st = time.time()
    logger = pl.loggers.TensorBoardLogger('tb_logs', name=f'option{option}', version=f'alpha_{alpha}')
    model = ResNet_KD(student_model, T, T1, T2, alpha, K, option, batch)
    trainer = pl.Trainer(gpus=-1, max_epochs=epochs, weights_summary=None, logger=logger)
    trainer.fit(model)
    print(f'Total training time: {(time.time() - st) / 60: .1f}m\n'
          f'Best Top1 Acc: {model.best_acc1.item() * 100:.2f}% - epoch : {model.best_epoch}\n'
          f'Best Top5 Acc: {model.best_acc5.item() * 100:.2f}%\n')

    f = open("log.txt", 'a')
    f.write('============================================================\n'
            'CONFIG \n'
            f'student_model = {student_model} | option = {option} | alpha = {alpha}\n'
            f'Total training time: {(time.time() - st) / 60: .1f}m\n'
            f'Best Top1 Acc: {model.best_acc1.item() * 100:.2f}% - epoch : {model.best_epoch}\n'
            f'Best Top5 Acc: {model.best_acc5.item() * 100:.2f}%\n')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-student', type=int)
    parser.add_argument('-t', type=float)
    parser.add_argument('-t1', type=float)
    parser.add_argument('-t2', type=float)
    parser.add_argument('-alpha', type=float)
    parser.add_argument('-k', type=int)
    parser.add_argument('-option', type=int)

    parser.add_argument('-batch', type=int)
    parser.add_argument('-epochs', type=int)

    args = parser.parse_args()
    kd_train_and_test(args.student,
                      args.t, args.t1, args.t2, args.alpha, args.k, args.option,
                      args.batch, args.epochs)
