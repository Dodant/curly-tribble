## Generator Network
# Input -> Conv(k9n64s1) -> PReLU ->
# Residual Block  - 5 blocks
# : Conv(k3n64s1) -> BN -> PReLU -> Conv(k3n64s1) -> BN -> Elmt Sum
# Conv(k3n64s1) -> BN -> Elmt Sum ->
# (Conv(k3n256s1) -> PixelShuffler * 2 -> PReLU) * 2 -> Conv(k9n3s1) -> SR!

## Discriminator Network
# Input -> Conv(k2n64s1) -> LeakyReLU
# Block - 7 blocks
# : Conv -> BN -> LeakyReLU
# k3n64s2 /
# k3n128s1 / k3n128s2 /
# k3n256s1 / k3n256s2 /
# k3n512s1 / k3n512s2 /
# Dense (1024) -> LeakReLU -> Dense (1) -> Sigmoid

import torch
import torch.nn as nn
import torch.nn.functional as F


class G_ResBlock(nn.Module):
    def __init__(self, chnl):
        super(G_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(chnl, chnl, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(chnl, chnl, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(chnl)
        self.prelu = nn.PReLU(num_parameters=chnl)

    def forward(self, x):
        x_ = self.prelu(self.bn(self.conv1(x)))
        x_ = self.bn(self.conv2(x_))
        return x + x_


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

        trunk = []
        for _ in range(16):
            trunk.append(G_ResBlock(64))
        self.trunk = nn.Sequential(*trunk)

        self.mid_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.upsample1 = G_UpsampleBlock(64)
        self.upsample2 = G_UpsampleBlock(64)
        self.last_conv = nn.Conv2d(64, 3, 9, padding=4)
        # self.tanh = nn.Tanh()

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
        self.cnn1 = nn.Conv2d(in_chnls, out_chnls, stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chnls)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.cnn1(x)))


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


# 4x upsampling
def test():
    low_resolution = 24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        generator = Generator()
        discriminator = Discriminator()
        gen_output = generator(x)
        dis_output = discriminator(x)

        print(gen_output.shape)
        print(dis_output.shape)