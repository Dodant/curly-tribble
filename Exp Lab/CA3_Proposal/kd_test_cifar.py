import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR100

import argparse

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

def kd_test(student_model, batch):
    # hyperparameter
    num_workers = 4

    student = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar100_resnet{student_model}", pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student.to(device)

    testset = CIFAR100(root='./data', train=False, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True)

    correct_s, correct_5s, total_s = 0, 0, 0

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


    print(f'Top1 Acc : - {correct_s * 100 / total_s:0.1f}% / Top5 Acc : student - {correct_5s * 100 / total_s:0.1f}%')
    print('===========================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int)
    parser.add_argument('-batch', type=int)

    args = parser.parse_args()
    kd_test(args.s, args.batch)
