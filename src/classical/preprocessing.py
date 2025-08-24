import cv2
import numpy as np

def to_gray_blur(img_bgr, ksize=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return blur

def region_of_interest(img, vertices=None):
    h, w = img.shape[:2]
    if vertices is None:
        # A trapezoid ROI typical for front-facing dashcam
        vertices = np.array([
            [(int(0.1*w), h),
             (int(0.45*w), int(0.6*h)),
             (int(0.55*w), int(0.6*h)),
             (int(0.9*w), h)]
        ], dtype=np.int32)
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(img, mask)
    return masked
