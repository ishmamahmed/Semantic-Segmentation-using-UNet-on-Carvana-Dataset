import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True

def main():
    #--------------------------------------------------------
    '''
    Data Augmentation Pipeline:
    1. First of all, we define the train and validation transforms.
    2. Then we pass this function to the dataset class.
    '''
    '''Augment the data using the albumentation library'''
    train_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                 A.Rotate(limit=35, p=1.0),
                                 A.HorizontalFlip(p=0.5),
                                 A.VerticalFlip(p=0.1),
                                 A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[1.0, 1.0, 1.0],
                                             max_pixel_value=255.0),
                                 ToTensorV2()
                                ])

    val_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                               A.Normalize(mean=[0.0, 0.0, 0.0],
                                           std=[1.0, 1.0, 1.0],
                                           max_pixel_value=255.0),
                               ToTensorV2()
                               ])
    
    '''get_loaders() function'''
    train_ds = CarvanaDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    val_ds = CarvanaDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)


    #--------------------------------------------------------
    # 2. Create the Model, define the Loss Function and the Optimizer.
    # model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # 3. Start the training
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(">>Batch Index: ", batch_idx)
            data = data.to(device=DEVICE)  # [8, 3, 572, 572]
            targets = targets.float().unsqueeze(1).to(device=DEVICE)    # [8,1,572,572]
            print("Data shape: ", data.shape)
            print("Target shape: ", targets.shape)
            # forward
            predictions = model(data)       # [8,1,388,388]
            print("Prediction shape: ", predictions.shape)      




if __name__ == '__main__':
    main()