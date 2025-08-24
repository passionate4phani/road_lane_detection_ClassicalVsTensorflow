import cv2
import numpy as np

def draw_lines(img, lines, color=(0,255,0), thickness=6):
    if lines is None:
        return img
    img_out = img.copy()
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_out, (x1,y1), (x2,y2), color, thickness)
    return img_out

def overlay_mask(base_bgr, mask, color=(0,255,0), alpha=0.4):
    if mask.ndim == 2:
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask3 = mask
    color_layer = np.zeros_like(base_bgr)
    color_layer[mask>0] = color
    return cv2.addWeighted(base_bgr, 1.0, color_layer, alpha, 0)
