import os
import torch
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from ..classical.canny_hough import process_frame_classical


# ---------- Load Keras Model ----------
def load_keras(weights_path="models/best_model.keras"):
    if not os.path.exists(weights_path):
        print("[INFO] No Keras model found; falling back to classical.")
        return None

    try:
        # Register dummy dice_loss so TF can load
        def dice_loss(*args, **kwargs):
            return None

        model = keras.models.load_model(
            weights_path,
            custom_objects={"dice_loss": dice_loss},  # Bypass custom loss
            compile=False
        )
        print(f"[INFO] Loaded Keras model from {weights_path}")
        return model
    except Exception as e:
        print(f"[WARN] Failed to load Keras model: {e}")
        return None


# ---------- Segmentation with Keras ----------
def segment_frame_keras(model, frame, target_size=(224, 224), thresh=0.5):
    """
    Run segmentation on a single frame using Keras model.

    Args:
        model: Loaded Keras model
        frame: Input frame (H, W, 3)
        target_size: Input size expected by the model
        thresh: Threshold for binary mask

    Returns:
        overlay: Overlayed segmentation result resized to original frame size
        mask: Binary mask resized to original frame size
    """
    orig_h, orig_w = frame.shape[:2]

    # Resize frame to model input
    img_resized = cv2.resize(frame, target_size)
    img_resized = img_resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(img_resized, axis=0)  # (1, 224, 224, 3)

    # Forward pass
    prob = model.predict(tensor, verbose=0)[0, :, :, 0]

    # Threshold -> binary mask
    mask_resized = (prob > thresh).astype(np.uint8) * 255

    # Resize mask back to original size
    mask = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Overlay mask on original frame
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)  # Green lanes

    return overlay, mask