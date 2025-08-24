import torch

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    num = 2. * (pred*target).sum() + smooth
    den = pred.sum() + target.sum() + smooth
    return 1 - num/den
