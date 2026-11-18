# Step 9: final image-level evaluation on TEST with the threshold frozen on validation
import os
import numpy as np
import matplotlib.pyplot as plt

th = float(np.load("patches, features and scores (numpy arrays)/threshold.npy")[0])
d = np.load("patches, features and scores (numpy arrays)/scores_test.npz")
names = d["names"]
img_score = d["scores"].max(axis=1)
y = np.array([os.path.getsize(f"SARD1/test/labels/{os.path.splitext(n)[0]}.txt") > 0 for n in names]).astype(int)
pred = (img_score >= th).astype(int)

tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
rec, prec = tp / (tp + fn), tp / (tp + fp)
f1 = 2 * prec * rec / (prec + rec)
f2 = 5 * prec * rec / (4 * prec + rec)
print(f"TEST, threshold {th:.3f} (frozen from validation): {len(y)} images, {y.sum()} with humans")
print(f"accuracy {(tp+tn)/len(y):.3f}  recall {rec:.3f}  precision {prec:.3f}  F1 {f1:.3f}  F2 {f2:.3f}")
print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}   false alarms on empty images {fp/(fp+tn):.1%}")

# list of mistakes for the error analysis
np.savez("patches, features and scores (numpy arrays)/test_errors.npz", fn=names[(pred == 0) & (y == 1)], fp=names[(pred == 1) & (y == 0)],
         scores=img_score, y=y)

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
for (i, j), v in {(0, 0): tn, (0, 1): fp, (1, 0): fn, (1, 1): tp}.items():
    ax.text(j, i, v, ha="center", va="center", fontsize=16)
ax.set_xticks([0, 1]); ax.set_xticklabels(["no human", "human"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["no human", "human"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Test confusion matrix (th {th:.2f})")
plt.tight_layout(); plt.savefig("charts/chart_test_confusion.png", dpi=150)
