# Step 8b: image-level decision. Image score = highest window score.
# Pick the threshold on VALIDATION images only.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

TARGET_RECALL = 0.95

d = np.load("patches, features and scores (numpy arrays)/scores_valid.npz")
names = d["names"]
img_score = d["scores"].max(axis=1)
# image is HUMAN if its label file has at least one box
y = np.array([os.path.getsize(f"SARD1/valid/labels/{os.path.splitext(n)[0]}.txt") > 0 for n in names]).astype(int)
print(f"{len(y)} validation images: {y.sum()} with humans, {(1-y).sum()} without")


def metrics(th):
    pred = (img_score >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    rec = tp / (tp + fn); prec = tp / max(tp + fp, 1)
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0
    return dict(th=th, acc=(tp + tn) / len(y), recall=rec, precision=prec, f2=f2, TP=tp, FP=fp, FN=fn, TN=tn,
                fpr=fp / (fp + tn))


# thresholds to try: every distinct image score
cands = np.unique(img_score)
table = [metrics(t) for t in cands]

# highest threshold that still reaches the target recall (= fewest false alarms for that recall)
ok = [m for m in table if m["recall"] >= TARGET_RECALL]
best = max(ok, key=lambda m: m["th"])
print(f"\nThreshold for recall >= {TARGET_RECALL:.0%}: {best['th']:.3f}")
for k in ("acc", "recall", "precision", "f2", "fpr", "TP", "FP", "FN", "TN"):
    print(f"  {k:9s} {best[k]:.3f}" if isinstance(best[k], float) else f"  {k:9s} {best[k]}")

print("\nOther operating points:")
for target in (0.90, 0.95, 0.98, 0.99):
    o = [m for m in table if m["recall"] >= target]
    m = max(o, key=lambda m: m["th"])
    print(f"  recall >= {target:.0%}: threshold {m['th']:.3f}, recall {m['recall']:.3f}, "
          f"precision {m['precision']:.3f}, false alarms on empty images {m['fpr']:.1%}")
m = metrics(0.5)
print(f"  default 0.5   : recall {m['recall']:.3f}, precision {m['precision']:.3f}, "
      f"false alarms on empty images {m['fpr']:.1%}")

np.save("patches, features and scores (numpy arrays)/threshold.npy", np.array([best["th"]]))

p, r, _ = precision_recall_curve(y, img_score)
fpr, tpr, _ = roc_curve(y, img_score)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(r, p); ax[0].scatter([best["recall"]], [best["precision"]], color="red", zorder=3,
                                label=f"chosen (recall {best['recall']:.2f})")
ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].set_title(f"Image-level PR curve (AP {average_precision_score(y, img_score):.3f})")
ax[0].legend()
ax[1].plot(fpr, tpr); ax[1].scatter([best["fpr"]], [best["recall"]], color="red", zorder=3)
ax[1].set_xlabel("False alarm rate (empty images flagged)"); ax[1].set_ylabel("Recall")
ax[1].set_title("Recall vs false-alarm rate")
plt.tight_layout()
plt.savefig("charts/chart_val_image_level.png", dpi=150)
print("saved threshold.npy and chart_val_image_level.png")
