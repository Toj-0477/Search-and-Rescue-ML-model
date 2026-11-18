# Helpers shared by the app and the demo scripts: resize an image, pick the boxes to show, draw them.
import numpy as np
import cv2


def letterbox(img, size=640):
    """Scale the longer side to 640 and pad to a square, the size the model was trained on."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    ph, pw = size - img.shape[0], size - img.shape[1]
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)


def fit_to_square(img, size=640):
    """Like letterbox, but also returns the size of the real (unpadded) picture inside the square,
    so windows can be kept off the padding. Padding is not real scenery: the model must never scan it."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    vw, vh = int(round(w * s)), int(round(h * s))
    sq = np.zeros((size, size, 3), np.uint8)
    sq[:vh, :vw] = cv2.resize(img, (vw, vh))
    return sq, vw, vh


def windows_inside(wins, vw, vh):
    """Keep only the windows that lie completely inside the real picture."""
    return wins[(wins[:, 2] <= vw) & (wins[:, 3] <= vh)]


def overlap(a, b):
    """Shared area divided by the SMALLER box's area, so a small window inside a big one counts as a duplicate."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))


def top_boxes(wins, scores, threshold, max_boxes=5):
    """Windows above the threshold, best first, dropping windows that overlap one already kept."""
    keep = []
    for i in np.argsort(-scores):
        if scores[i] < threshold or len(keep) >= max_boxes:
            break
        if all(overlap(wins[i], wins[j]) < 0.2 for j in keep):
            keep.append(i)
    return keep


def draw_result(img, wins, scores, threshold):
    """Returns (best score, flagged?, RGB image with red boxes on the likely people)."""
    best = float(scores.max())
    flagged = best >= threshold
    out = img.copy()
    if flagged:
        for i in top_boxes(wins, scores, threshold):
            x1, y1, x2, y2 = wins[i]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, f"{scores[i]:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return best, flagged, cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
