import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import UNet
from .dataset import LaneDataset
from .utils import dice_loss

def train(
    images_dir="path/to/train/images",
    masks_dir="path/to/train/masks",
    epochs=10, lr=1e-3, batch_size=4, save_path="models/unet_lane.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LaneDataset(images_dir, masks_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    model = UNet(n_channels=3, n_classes=1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        lossv = 0.0
        for img, m in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            img, m = img.to(device), m.to(device)
            logits = model(img)
            loss = dice_loss(logits, m)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lossv += loss.item()
        print(f"Epoch {ep}: loss={lossv/len(dl):.4f}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

if __name__ == "__main__":
    train()
