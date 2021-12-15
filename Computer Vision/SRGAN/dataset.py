import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ResDataset(Dataset):
    def __init__(self, path, hr_shape):
        hr_h, hr_w = hr_shape

        self.lr_transform = transforms.Compose(
            [transforms.Resize((hr_h // 4, hr_w // 4), Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.hr_transform = transforms.Compose(
            [transforms.Resize((hr_h, hr_w), Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.files = sorted(glob.glob(path + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)