import os, glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class LaneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*")))
        assert len(self.images) == len(self.masks), "Images and masks count mismatch"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")  # single channel
        img = np.array(img)
        mask = (np.array(mask) > 127).astype(np.float32)
        if self.transform:
            # You can integrate Albumentations here if needed
            pass
        # to tensor
        img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask
