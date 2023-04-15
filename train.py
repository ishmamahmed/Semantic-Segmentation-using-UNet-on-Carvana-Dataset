import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from dataset import CarvanaDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNet
from torch.utils.tensorboard import SummaryWriter
from utils import check_accuracy, save_predictions_as_imgs

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
IMAGE_HEIGHT = 512 # 160  # 1280 originally
IMAGE_WIDTH = 512 # 240  # 1918 originally
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
# model_path = 'models/train_02_d = '+str(discount)+'.pt'
model_path = "my_checkpoint.pth.tar"
lossLog = SummaryWriter()

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
    loss_fn = nn.CrossEntropyLoss()      # NEED TO CHECK (1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        print("--- Training from the last checkpoint")
        checkpoint = torch.load(model_path)
        
        model.load_state_dict(checkpoint["state_dict"])
        current_epoch = checkpoint["epoch"]
        # seed = checkpoint['seed']

    else:
        print("--- Training from beginning")
        current_epoch = 0


    # 3. Start the training
    for epoch in tqdm(range(current_epoch, NUM_EPOCHS)):
        loss_per_epoch = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            
            data = data.to(device=DEVICE)  # [8, 3, 572, 572]
            targets = targets.float().unsqueeze(1).to(device=DEVICE)  # [8, 1, 572, 572]

            # forward
            predictions = model(data)       # [8,1,388,388]
            loss = loss_fn(predictions, targets)    # Need to check this line
            loss_per_epoch += loss
            # backward
            optimizer.zero_grad()
            loss.backward()

            # update the model parameters
            optimizer.step()

        # Saving the model
        print("=> Saving Checkpoint")
        checkpoint = {"model": model.state_dict(),
                      "optimizer":optimizer.state_dict(),
                      "epoch": epoch,
                      # "seed": get_seed
                      }
        torch.save(checkpoint, model_path)

        # Save the data to tensorboard
        lossLog.add_scalar("Training loss per epoch", loss, epoch)
        # Check accuracy
        dice_score = check_accuracy(val_loader, model, device=DEVICE)
        # Print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == '__main__':
    main()