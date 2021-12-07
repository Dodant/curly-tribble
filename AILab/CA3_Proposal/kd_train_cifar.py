import time
import argparse
from warnings import filterwarnings

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100


filterwarnings('ignore')

train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomCrop((32, 32), padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


def DistillationLoss(student_logit, teacher_logit, T):
    soft_label = F.softmax(teacher_logit / T, dim=1)
    soft_prediction = F.log_softmax(student_logit / T, dim=1)
    return F.kl_div(soft_prediction, soft_label)


def FinalLoss(teacher_logit, student_logit, labels, T, alpha):
    return (1. - alpha) * F.cross_entropy(student_logit, labels) \
           + (alpha * T * T) * DistillationLoss(student_logit, teacher_logit, T)


def kd_train(student_model, T, alpha, epochs, batch):
    # hyperparameter
    num_workers = 4

    # teacher = ResNet56
    # student = ResNet20, ResNet32, ResNet56
    model_link = "chenyaofo/pytorch-cifar-models"
    teacher = torch.hub.load(model_link, "cifar100_resnet56", pretrained=True)
    student = torch.hub.load(model_link, f"cifar100_resnet{student_model}", pretrained=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)
    teacher.eval()

    trainset = CIFAR100(root='./data', train=True, transform=train_transform)
    testset = CIFAR100(root='./data', train=False, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(student.parameters(), lr=0.001)

    print(f'Config : Student = ResNet{student_model}, T = {T}, alpha = {alpha}')

    train_st = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            student_logit = student(inputs)
            teacher_logit = teacher(inputs)
            loss = FinalLoss(teacher_logit, student_logit, labels, T, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(student_logit, 1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data)
            train_samples += len(inputs)

        epoch_loss = train_loss / len(trainloader)
        epoch_acc = train_acc.float() / train_samples * 100

        # print(f"epoch: {epoch + 1} || tl: {epoch_loss:.3f}, ta: {epoch_acc:.2f}%")

    print(f'Training Finished - Train time : {(time.time() - train_st) // 60}m\n')

    PATH = f'./student_cifar_{student_model}_{alpha}.pth'
    torch.save(student.state_dict(), PATH)

    correct_s, correct_5s, total_s = 0, 0, 0
    correct_t, correct_5t, total_t = 0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # student - top1
            outputs_s = student(images)
            _, predicted_s = torch.max(outputs_s.data, 1)
            total_s += labels.size(0)
            correct_s += (predicted_s == labels).sum().item()
            # top5
            for idx, item in enumerate(labels.view(-1, 1)):
                if item in torch.topk(outputs_s, 5).indices[idx]:
                    correct_5s += 1

            # teacher - top1
            outputs_t = teacher(images)
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted_t == labels).sum().item()
            # top5
            for idx, item in enumerate(labels.view(-1, 1)):
                if item in torch.topk(outputs_t, 5).indices[idx]:
                    correct_5t += 1

    print(f'Top1 Acc : student - {correct_s * 100 / total_s:0.1f}% / teacher - {correct_t * 100 / total_t:0.1f}%')
    print(f'Top5 Acc : student - {correct_5s * 100 / total_s:0.1f}% / teacher - {correct_5t * 100 / total_t:0.1f}%')
    print('===========================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int)
    parser.add_argument('-t', type=int)
    parser.add_argument('-alpha', type=float)
    parser.add_argument('-epochs', type=int)
    parser.add_argument('-batch', type=int)

    args = parser.parse_args()
    kd_train(args.s, args.t, args.alpha, args.epochs, args.batch)
