"""cv2 overlay primitives on a BGR uint8 image. Each draws in place + returns
the image. Non-finite / out-of-bounds points are skipped."""
import cv2
import numpy as np

def _pt(u, v):
    return (int(round(u)), int(round(v)))

def draw_points(img, uv, color, radius=3):
    for p in np.asarray(uv, float).reshape(-1, 2):
        if np.isfinite(p).all():
            cv2.circle(img, _pt(*p), radius, color, -1)
    return img

def draw_chain(img, uv_list, color, thickness=1):
    pts = [_pt(u, v) for (u, v) in uv_list if np.isfinite([u, v]).all()]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, thickness)
    for p in pts:
        cv2.circle(img, p, max(2, thickness + 1), color, -1)
    return img

def draw_cloud(img, uv, color, radius=1):
    return draw_points(img, uv, color, radius=radius)

def draw_mask(img, mask_bool, color, alpha=0.35, outline=True):
    m = np.asarray(mask_bool, bool)
    if m.any():
        ov = img.copy(); ov[m] = color
        img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
        if outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, color, 1)
    return img

def draw_axis(img, tail_uv, head_uv, color):
    if np.isfinite(tail_uv).all() and np.isfinite(head_uv).all():
        cv2.arrowedLine(img, _pt(*tail_uv), _pt(*head_uv), color, 2, tipLength=0.3)
    return img

def legend(img, items):
    for i, (text, color) in enumerate(items):
        cv2.putText(img, text, (5, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img
