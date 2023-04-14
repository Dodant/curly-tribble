import PIL.Image as Image

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from model import Generator

PATH = 'saved_models/generator_300.pth'
generator = Generator()
generator.load_state_dict(torch.load(PATH))
generator.eval()

transform = transforms.Compose(
    [transforms.Resize((128, 128), Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

set5 = datasets.ImageFolder('archive/set5/Set5/', transform=transform)
testloader = DataLoader(set5, batch_size=1, shuffle=True, drop_last=True)

for i, lr in enumerate(testloader):
    lr = torch.FloatTensor(lr[0])
    sr = generator(lr)
    lr = nn.functional.interpolate(lr, scale_factor=4)
    lr = make_grid(lr, nrow=1, normalize=True)
    sr = make_grid(sr, nrow=1, normalize=True)
    img_grid = torch.cat((lr, sr), -1)
    save_image(img_grid, f"set5_{i}.png", normalize=False)
