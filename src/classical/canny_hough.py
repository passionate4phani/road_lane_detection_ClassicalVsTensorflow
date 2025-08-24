import cv2
import numpy as np
from .preprocessing import to_gray_blur, region_of_interest
from .utils import draw_lines, overlay_mask

def process_frame_classical(frame_bgr, return_debug=False):
    """
    Returns overlay (BGR) and optionally an edge debug image.
    """
    h, w = frame_bgr.shape[:2]
    blur = to_gray_blur(frame_bgr, ksize=5)
    edges = cv2.Canny(blur, 50, 150)
    roi_edges = region_of_interest(edges)

    # Probabilistic Hough Transform
    lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=int(0.1*w), maxLineGap=50)

    overlay = frame_bgr.copy()
    if lines is not None:
        overlay = draw_lines(overlay, lines, color=(0,255,0), thickness=6)

    if return_debug:
        dbg = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
        return overlay, dbg
    else:
        return overlay, None
