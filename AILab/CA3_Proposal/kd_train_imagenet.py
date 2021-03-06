import time
import argparse
from warnings import filterwarnings

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import models, transforms, datasets

filterwarnings('ignore')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5)
])

datasets_path = './imagenet/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train'


def DistillationLoss(student_logit, teacher_logit, T):
    soft_label = F.softmax(teacher_logit / T, dim=1)
    soft_prediction = F.log_softmax(student_logit / T, dim=1)
    return F.kl_div(soft_prediction, soft_label)


def FinalLoss(teacher_logit, student_logit, labels, T, alpha):
    return (1. - alpha) * F.cross_entropy(student_logit, labels) \
           + (alpha * T * T) * DistillationLoss(student_logit, teacher_logit, T)


def kd_train(student_model, T, alpha, epochs, batch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # teacher = ResNet50
    teacher = models.resnet50(pretrained=True).to(device).eval()

    # student = ResNet18, ResNet34, ResNet50
    student = 0
    if student_model == 18:
        student = models.resnet18(pretrained=False).to(device)
    elif student_model == 34:
        student = models.resnet34(pretrained=False).to(device)
    elif student_model == 50:
        student = models.resnet50(pretrained=False).to(device)

    imagenet = datasets.ImageFolder(datasets_path, transform=transform)
    trainset, testset = random_split(imagenet, [1_000_000, len(imagenet) - 1_000_000])
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scaler = amp.GradScaler()

    print(f'Config : Student = ResNet{student_model}, T = {T}, alpha = {alpha}')

    train_st = time.time()
    for epoch in range(epochs):
        train_loss, train_acc, train_samples = 0.0, 0.0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with amp.autocast():
                student_logit = student(inputs)
                teacher_logit = teacher(inputs)
                loss = FinalLoss(teacher_logit, student_logit, labels, T, alpha)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(student_logit, 1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data)
            train_samples += len(inputs)

        epoch_loss = train_loss / len(trainloader)
        epoch_acc = train_acc.float() / train_samples * 100
        print(f"epoch: {epoch + 1:>2d} || tl: {epoch_loss:.3f}, ta: {epoch_acc:.1f}%")

    print(f'Training Finished - Train time : {(time.time() - train_st) // 60}m\n')
    torch.save(student.state_dict(), f'./student_{student_model}_{T}.pth')

    correct_s, correct_5s, correct_10s, total_s = 0, 0, 0, 0
    correct_t, correct_5t, correct_10t, total_t = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # student - top1
            outputs_s = student(images)
            _, predicted_s = torch.max(outputs_s.data, 1)
            total_s += labels.size(0)
            correct_s += (predicted_s == labels).sum().item()
            # top5, top10
            for idx, item in enumerate(labels.view(-1, 1)):
                if item in torch.topk(outputs_s, 5).indices[idx]:
                    correct_5s += 1
                    correct_10s += 1
                elif item in torch.topk(outputs_s, 10).indices[idx]:
                    correct_10s += 1

            # teacher - top1
            outputs_t = teacher(images)
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted_t == labels).sum().item()
            # top5, top10
            for idx, item in enumerate(labels.view(-1, 1)):
                if item in torch.topk(outputs_t, 5).indices[idx]:
                    correct_5t += 1
                    correct_10t += 1
                elif item in torch.topk(outputs_t, 10).indices[idx]:
                    correct_10t += 1

    print(f'Top1 Acc : student - {correct_s * 100 / total_s:.1f}% / teacher - {correct_t * 100 / total_t:.1f}%')
    print(f'Top5 Acc : student - {correct_5s * 100 / total_s:.1f}% / teacher - {correct_5t * 100 / total_t:.1f}%')
    print(f'Top10 Acc : student - {correct_10s * 100 / total_s:.1f}% / teacher - {correct_10t * 100 / total_t:.1f}%')
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
