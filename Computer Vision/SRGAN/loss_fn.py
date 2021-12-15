import torch.cuda
import torch.nn as nn
from torchvision.models import vgg19


class VGG_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()
        if torch.cuda.is_available():
            self.vgg = vgg19(pretrained=True).features[:36].eval().cuda()
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input_, target):
        return self.loss(self.vgg(input_), self.vgg(target))