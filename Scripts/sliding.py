# Sliding window helpers used by the later steps
import numpy as np
import cv2
from step3_features import hog_feat, color_feat, lbp_feat

W, H = 64, 128                 # patch size the model was trained on
SCALES = (64, 96, 128, 176)    # window heights in px (width = height / 2)


def make_windows(scales=SCALES, stride_frac=0.5, size=640):
    """All windows as (x1, y1, x2, y2). Stride = stride_frac * window height."""
    wins = []
    for h in scales:
        w = h // 2
        stride = max(8, int(h * stride_frac))
        for y in range(0, size - h + 1, stride):
            for x in range(0, size - w + 1, stride):
                wins.append((x, y, x + w, y + h))
    return np.array(wins)


def window_features(img, wins):
    """F4 features (HOG + colour + LBP) for every window, in the same order as training."""
    feats = []
    for x1, y1, x2, y2 in wins:
        p = cv2.resize(img[y1:y2, x1:x2], (W, H))
        feats.append(np.concatenate([hog_feat(p), color_feat(p), lbp_feat(p)]))
    return np.array(feats)


def score_image(img, scaler, model, wins):
    """Probability of 'human' for every window."""
    X = scaler.transform(window_features(img, wins))
    return model.predict_proba(X)[:, 1]
