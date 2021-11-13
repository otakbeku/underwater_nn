import torch
import torch.nn as nn
import torch.nn.functional as F

import os


# Dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

from PIL import Image


# ignore warning
import warnings
warnings.filterwarnings('ignore')


class T_CNN(nn.Module):
    def __init__(self,
                 ) -> None:
        super(T_CNN, self).__init__()

        self.conv_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_2 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv_3 = nn.Conv2d(16, 16, 3, padding=1)
        # Concatenate
        self.conv_4 = nn.Conv2d((16+16+16+3), 16, 3, padding=1)
        # self.conv_5 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv_6 = nn.Conv2d(16, 16, 3, padding=1)
        # Concatenate
        self.conv_7 = nn.Conv2d(((16+16+16+3)+16+16+16), 16, 3, padding=1)
        # self.conv_8 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv_9 = nn.Conv2d(16, 16, 3, padding=1)
        # Concatenate
        self.conv_10 = nn.Conv2d(147, 3, 3, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.conv_1(x))  # 1
        x2 = F.relu(self.conv_2(x1))  # 2
        x3 = F.relu(self.conv_2(x2))  # 3
        x_concat1 = torch.cat((x1, x2, x3, x), dim=1)
        x4 = F.relu(self.conv_4(x_concat1))  # 4
        x5 = F.relu(self.conv_2(x1))  # 5
        x6 = F.relu(self.conv_2(x2))  # 6
        x_concat2 = torch.cat((x_concat1, x4, x5, x6), dim=1)
        x7 = F.relu(self.conv_7(x_concat2))  # 7
        x8 = F.relu(self.conv_2(x1))  # 8
        x9 = F.relu(self.conv_2(x2))  # 9
        x_concat3 = torch.cat((x_concat2, x7, x8, x9), dim=1)
        x_out = self.conv_10(x_concat3)
        x_out += x
        return x_out


class Paired_Dataset(Dataset):
    def __init__(self, train_folder, target_folder, transform=None):
        self.train_folder = train_folder
        self.target_folder = target_folder
        self.image_list = os.listdir(self.train_folder)
        print(f'Number of Images: {len(self.image_list)}')
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        train_image = Image.open(os.path.join(
            self.train_folder, self.image_list[index]))
        target_image = Image.open(os.path.join(
            self.target_folder, self.image_list[index]))
        if self.transform:
            train_image = self.transform(train_image)
            target_image = self.transform(target_image)
        return train_image, target_image
