# Step 2: cut the images into small patches (HUMAN = 1, NOT HUMAN = 0)
# From step 1: boxes are mostly tall (median 36x82 px), so patches are 64 wide x 128 tall.
import os
import numpy as np
import cv2

ROOT = "SARD1"
W, H = 64, 128
MARGIN = 0.15      # extra context around each box
NEG_RATIO = 4      # negatives per positive
rng = np.random.default_rng(42)


def read_boxes(label_path):
    boxes = []
    for line in open(label_path).read().strip().splitlines():
        if line.strip():
            _, cx, cy, w, h = map(float, line.split()[:5])
            boxes.append([(cx - w/2) * 640, (cy - h/2) * 640, (cx + w/2) * 640, (cy + h/2) * 640])
    return np.array(boxes).reshape(-1, 4)


def crop(img, cx, cy, ch):
    """Crop a window of height ch (width ch/2) centred on (cx, cy), resize to 64x128.
    Parts outside the image are filled by repeating the edge pixels."""
    cw = ch / 2
    x1, y1 = int(round(cx - cw/2)), int(round(cy - ch/2))
    x2, y2 = int(round(cx + cw/2)), int(round(cy + ch/2))
    pad = max(0, -x1, -y1, x2 - img.shape[1], y2 - img.shape[0])
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return cv2.resize(img[y1:y2, x1:x2], (W, H))


def iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter)


for split in ["train", "valid", "test"]:
    img_dir, lab_dir = f"{ROOT}/{split}/images", f"{ROOT}/{split}/labels"
    patches, labels = [], []
    for name in sorted(os.listdir(img_dir)):
        img = cv2.imread(f"{img_dir}/{name}")
        boxes = read_boxes(f"{lab_dir}/{os.path.splitext(name)[0]}.txt")

        # positives
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            ch = max(bh, bw * 2) * (1 + 2 * MARGIN)
            patches.append(crop(img, (x1 + x2) / 2, (y1 + y2) / 2, ch))
            labels.append(1)

        # negatives: random windows that barely overlap any person (IoU < 0.1)
        n_neg = NEG_RATIO * max(1, len(boxes))
        got, tries = 0, 0
        while got < n_neg and tries < 50:
            tries += 1
            ch = rng.uniform(48, 170)               # window heights similar to person heights
            cx = rng.uniform(ch / 4, 640 - ch / 4)
            cy = rng.uniform(ch / 2, 640 - ch / 2)
            win = np.array([cx - ch/4, cy - ch/2, cx + ch/4, cy + ch/2])
            if iou(win, boxes).max(initial=0) < 0.1:
                patches.append(crop(img, cx, cy, ch))
                labels.append(0)
                got += 1

    X = np.array(patches, dtype=np.uint8)
    y = np.array(labels, dtype=np.uint8)
    np.savez_compressed(f"patches, features and scores (numpy arrays)/patches_{split}.npz", X=X, y=y)
    print(f"{split}: {len(y)} patches, {y.sum()} human, {(y == 0).sum()} not human")
