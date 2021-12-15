import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from loss_fn import *
from model import *
from dataset import *

EPOCHS = 1000
SR_SHAPE = 256

cuda = torch.cuda.is_available()

generator = Generator()
discriminator = Discriminator()

content_loss = VGG_loss()
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

dataloader = DataLoader(
    ResDataset('./DIV2K_train_HR/Train', hr_shape=(SR_SHAPE, SR_SHAPE)),
    batch_size=4,
    shuffle=True,
    drop_last=True
)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    content_loss = content_loss.cuda()
    adversarial_loss = adversarial_loss.cuda()

print('go')
total_st = time.time()
for epoch in range(EPOCHS):
    st = time.time()
    train_loss = 0.0

    for i, images in enumerate(dataloader):
        lr = Variable(images["lr"].type(torch.cuda.FloatTensor))
        hr = Variable(images["hr"].type(torch.cuda.FloatTensor))
        sr = generator(lr)

        valid = Variable(torch.cuda.FloatTensor(np.ones((lr.size(0), 1))), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(np.zeros((lr.size(0), 1))), requires_grad=False)

        ######################
        # train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(hr), valid)
        fake_loss = adversarial_loss(discriminator(sr.detach()), fake)
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        ######################
        # train Generator
        optimizer_G.zero_grad()
        # Total loss = Content loss + Adversarial loss
        loss_G = content_loss(sr, hr) + 1e-3 * adversarial_loss(discriminator(sr), valid).detach()
        loss_G.backward()
        optimizer_G.step()

        train_loss += loss_G.item() + loss_D.item() # 합치는 거 의미 없음

    epoch_loss = train_loss / len(dataloader)

    lr = nn.functional.interpolate(lr, scale_factor=4)
    sr = make_grid(sr, nrow=1, normalize=True)
    lr = make_grid(lr, nrow=1, normalize=True)
    img_grid = torch.cat((lr, sr), -1)
    save_image(img_grid, f"images/epoch_{epoch}_16.png", normalize=False)

    print(f'Epoch: {epoch:>3d}  loss: {epoch_loss:.3f}  --  {(time.time()-st)/60:.1f}m')

    if (epoch+1) % 10 == 0:
        torch.save(generator.state_dict(), f"saved_models/generator_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_{epoch+1}.pth")

print(f'Finish -- total training time {(time.time()-total_st)/60:.1f}m')