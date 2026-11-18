# Step 1: look at the dataset (class balance, box sizes, label sanity, duplicates)
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

ROOT = "SARD1"
SPLITS = ["train", "valid", "test"]

all_w, all_h = [], []
base_names = {}  # split -> set of original image names (before the roboflow hash)

for s in SPLITS:
    lab_dir = f"{ROOT}/{s}/labels"
    files = os.listdir(lab_dir)
    n_empty, n_boxes, bad = 0, 0, 0
    boxes_per_img = []
    for f in files:
        rows = [l.split() for l in open(f"{lab_dir}/{f}").read().strip().splitlines() if l.strip()]
        if not rows:
            n_empty += 1
        boxes_per_img.append(len(rows))
        for r in rows:
            cx, cy, w, h = map(float, r[1:5])
            n_boxes += 1
            # sanity: zero-size or outside the image
            if w <= 0 or h <= 0 or cx - w/2 < -0.01 or cy - h/2 < -0.01 or cx + w/2 > 1.01 or cy + h/2 > 1.01:
                bad += 1
            if s == "train":
                all_w.append(w * 640)
                all_h.append(h * 640)
    base_names[s] = {f.split("_jpg.rf.")[0] for f in files}
    print(f"{s}: {len(files)} images, {n_empty} empty ({n_empty/len(files):.1%}), "
          f"{n_boxes} boxes, max {max(boxes_per_img)} per image, {bad} bad boxes")

# image size check on one image
img = cv2.imread(f"{ROOT}/train/images/" + os.listdir(f"{ROOT}/train/images")[0])
print("image shape:", img.shape)

# duplicates: same original photo showing up in more than one split?
print("\nOriginal photos shared between splits (leakage check):")
print("train & valid:", len(base_names["train"] & base_names["valid"]))
print("train & test :", len(base_names["train"] & base_names["test"]))
print("valid & test :", len(base_names["valid"] & base_names["test"]))
n_train_files = len(os.listdir(f"{ROOT}/train/labels"))
print(f"train has {n_train_files} files but only {len(base_names['train'])} original photos")

# box size stats (pixels, 640x640 image)
w, h = np.array(all_w), np.array(all_h)
print("\nBox width  px: min %.0f median %.0f mean %.0f max %.0f" % (w.min(), np.median(w), w.mean(), w.max()))
print("Box height px: min %.0f median %.0f mean %.0f max %.0f" % (h.min(), np.median(h), h.mean(), h.max()))
print("Percentiles of longer side (10/25/50/75/90):", np.percentile(np.maximum(w, h), [10, 25, 50, 75, 90]).round())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].hist(w, bins=50); ax[0].set_title("Box width (px)")
ax[1].hist(h, bins=50); ax[1].set_title("Box height (px)")
ax[2].hist(w / h, bins=50, range=(0, 5)); ax[2].set_title("Aspect ratio (w/h)")
plt.tight_layout()
plt.savefig("charts/box_sizes.png")
print("saved box_sizes.png")
