## Generator Network
# Input -> Conv -> PReLU ->
# Residual Block - B blocks
# : Conv -> BN -> PReLU -> Conv -> BN -> Elmt Sum
# Conv -> BN -> Elmt Sum ->
# Upsampling Block - 2 blocks
# : Conv -> PixelShuffler * 2 -> PReLU
# Conv -> SR!

## Discriminator Network
# Input -> Conv -> LeakyReLU
# Block - 7 blocks
# : Conv -> BN -> LeakyReLU
# Dense(1024) -> LeakyReLU -> Dense(1) -> Sigmoid

import torch.nn as nn


class G_ResBlock(nn.Module):
    def __init__(self, chnl):
        super(G_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(chnl, chnl, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(chnl, chnl, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(chnl)
        self.prelu = nn.PReLU(num_parameters=chnl)

    def forward(self, x):
        x_ = self.bn(self.conv2(self.prelu(self.bn(self.conv1(x)))))
        x = x + x_  # Elementwise sum
        return x


class G_UpsampleBlock(nn.Module):
    def __init__(self, in_chnls):
        super(G_UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_chnls, in_chnls*2**2, 3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU(num_parameters=in_chnls)

    def forward(self, x):
        return self.prelu(self.pixelshuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self): # 3 64
        super(Generator, self).__init__()
        self.init_conv = nn.Conv2d(3, 64, 9, padding=4)
        self.prelu = nn.PReLU()

        # src code form github.com/Lornatang/SRGAN-PyTorch/blob/master/model.py
        trunk = [G_ResBlock(64) for _ in range(16)]
        self.trunk = nn.Sequential(*trunk)

        self.mid_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.upsample1 = G_UpsampleBlock(64)
        self.upsample2 = G_UpsampleBlock(64)
        self.last_conv = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        b = self.prelu(self.init_conv(x))
        x = self.trunk(b)
        x = self.bn(self.mid_conv(x)) + b
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.last_conv(x)
        return x


class D_Block(nn.Module):
    def __init__(self, in_chnls, out_chnls, stride):
        super(D_Block, self).__init__()
        self.cnn = nn.Conv2d(in_chnls, out_chnls, stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chnls)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.cnn(x)))


class Discriminator(nn.Module):
    def __init__(self): # 3 64
        super(Discriminator, self).__init__()
        self.init_conv = nn.Conv2d(3, 64, 3, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.block1 = D_Block(64, 64, 2)
        self.block2 = D_Block(64, 128, 1)
        self.block3 = D_Block(128, 128, 2)
        self.block4 = D_Block(128, 256, 1)
        self.block5 = D_Block(256, 256, 2)
        self.block6 = D_Block(256, 512, 1)
        self.block7 = D_Block(512, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d((6,6))
        self.flat = nn.Flatten()
        self.fc1024 = nn.Linear(512*6*6, 1024)
        self.fc1 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.init_conv(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.flat(self.pool(x))
        x = self.leakyrelu(self.fc1024(x))
        x = self.sigmoid(self.fc1(x))
        return x
