import os
from PIL import Image
import numpy as np

class CarvanaDataset:
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # get list of all the images in the image directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

# current = os.getcwd()
# print("Current working Directory: ",current)
# path = os.path.join(current,os.listdir(current)[0])
# print(path)
# img_path = os.path.join(path,os.listdir(path)[5])
# print(img_path)
# img = Image.open(img_path).convert("RGB")
# img.show()
# print(img.size)

