# imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from dataset import CarvanaDataset

BATCH_SIZE = 8
IMAGE_HEIGHT = 512 # 160  # 1280 originally
IMAGE_WIDTH = 512 # 240  # 1918 originally
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
NUM_WORKERS = 2
PIN_MEMORY = True

def main():

    train_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                    # A.Rotate(limit=35, p=1.0),
                                    # # A.HorizontalFlip(p=0.5),
                                    # A.VerticalFlip(p=0.1),
                                    A.Normalize(mean=[0.0, 0.0, 0.0],
                                                std=[1.0, 1.0, 1.0],
                                                max_pixel_value=255.0),
                                    ToTensorV2()
                                ])

    train_ds = CarvanaDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)


    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    print(type(images[0]))
    img_numpy = (images[0].numpy().reshape(512,512,3)*255).astype(np.uint8)
    print(type(img_numpy), img_numpy.shape)
    img = Image.fromarray(img_numpy)
    # img.show()
    img_numpy = (labels[0].numpy().reshape(512,512,1)).astype(np.uint8)
    print(type(img_numpy), img_numpy.shape)
    img = Image.fromarray(img_numpy)
    img.show()

if __name__ == '__main__':
    main()