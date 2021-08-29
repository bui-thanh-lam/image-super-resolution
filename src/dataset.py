import os
import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset


class SRDataset(Dataset):

    def __init__(
        self, 
        HR_dir: str, 
        LR_dir: str, 
        metadata_file: str,
        mode: ImageReadMode
    ):
        with open(metadata_file, "r") as f:
            self.img_names = f.read().splitlines()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.mode = mode

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.LR_dir, self.img_names[idx])
        hr_img_path = os.path.join(self.HR_dir, self.img_names[idx])
        lr = torch.mul(1.0 / 255, read_image(lr_img_path, self.mode).float())
        hr = torch.mul(1.0 / 255, read_image(hr_img_path, self.mode).float())
        return lr, hr
