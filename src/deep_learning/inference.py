import os
import torch
import numpy as np
import cv2
from .model import UNet
from ..classical.canny_hough import process_frame_classical

def load_unet(weights_path="models/unet_lane.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
    print("[INFO] No weights provided; U-Net will fallback to classical mask for visualization.")
    model.eval()
    return model  # untrained (we will fallback to classical output)

def segment_frame_unet(model, frame_bgr, thresh=0.5):
    """
    Returns (overlay_bgr, mask_gray). If weights are not found, uses classical edges to form a mask.
    """
    device = next(model.parameters()).device
    h, w = frame_bgr.shape[:2]

    # Prepare tensor
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)

    # Try forward
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    # If model is untrained (all ~0.5), or no weights, the mask will be near-uniform.
    # Detect that and fallback to classical edges for a nicer demo mask.
    near_uniform = np.std(prob) < 1e-3

    if near_uniform:
        classical_overlay, _ = process_frame_classical(frame_bgr, return_debug=False)
        gray = cv2.cvtColor(classical_overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = (prob > thresh).astype(np.uint8)*255

    # simple morphology to clean
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    overlay = frame_bgr.copy()
    overlay[mask>0] = (0.6*overlay[mask>0] + 0.4*np.array([0,255,0])).astype(np.uint8)
    return overlay, mask
