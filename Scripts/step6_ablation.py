# Step 6: feature ablation - which features actually help?
# F1 = HOG, F2 = HOG + colour, F3 = HOG + LBP, F4 = HOG + colour + LBP
# Models: default Random Forest (main model) and RBF SVM (tuned params from step 5)
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, confusion_matrix

SEED = 42
tr, va = np.load("patches, features and scores (numpy arrays)/features_train.npz"), np.load("patches, features and scores (numpy arrays)/features_valid.npz")
ytr, yva = tr["y"], va["y"]

sets = {
    "F1 HOG": ["hog"],
    "F2 HOG+colour": ["hog", "color"],
    "F3 HOG+LBP": ["hog", "lbp"],
    "F4 HOG+colour+LBP": ["hog", "color", "lbp"],
}

rows = []
for fname, parts in sets.items():
    Xtr = np.hstack([tr[p] for p in parts])
    Xva = np.hstack([va[p] for p in parts])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    pca = PCA(n_components=150, random_state=SEED).fit(Xtr)
    Ptr, Pva = pca.transform(Xtr), pca.transform(Xva)

    for mname, model, a, b in [
        ("Random Forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=6, random_state=SEED), Xtr, Xva),
        ("RBF SVM", SVC(C=88.5, gamma=4.9e-5, kernel="rbf", class_weight="balanced"), Ptr, Pva),
    ]:
        t = time.time()
        model.fit(a, ytr)
        pred = model.predict(b)
        tn, fp, fn, tp = confusion_matrix(yva, pred).ravel()
        r = dict(features=fname, model=mname, n_features=Xtr.shape[1],
                 accuracy=accuracy_score(yva, pred), recall=recall_score(yva, pred),
                 precision=precision_score(yva, pred), f2=fbeta_score(yva, pred, beta=2), FN=fn)
        rows.append(r)
        print(f"{fname:20s} {mname:14s} acc {r['accuracy']:.3f} recall {r['recall']:.3f} "
              f"precision {r['precision']:.3f} F2 {r['f2']:.3f} FN {fn} ({time.time()-t:.0f}s)", flush=True)
        if fname.startswith("F4") and mname == "Random Forest":
            # keep this one (scaler + default forest) as the main model for the next steps
            joblib.dump((sc, model), "models/model_rf_default.joblib")

df = pd.DataFrame(rows)
df.to_csv("csvs/results_ablation.csv", index=False)

# grouped bar chart: recall and precision per feature set, one panel per model
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, mname in zip(axes, ["Random Forest", "RBF SVM"]):
    d = df[df.model == mname]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["recall"], 0.4, label="recall", color="steelblue")
    ax.bar(x + 0.2, d["precision"], 0.4, label="precision", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(d["features"], rotation=15)
    ax.set_title(mname)
    ax.set_ylim(0.6, 1.0)
    for i, (r_, p_) in enumerate(zip(d["recall"], d["precision"])):
        ax.text(i - 0.2, r_ + 0.004, f"{r_:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, p_ + 0.004, f"{p_:.3f}", ha="center", fontsize=8)
axes[0].legend()
plt.suptitle("Feature ablation (validation patches)")
plt.tight_layout()
plt.savefig("charts/chart_ablation.png", dpi=150)
print("saved results_ablation.csv, chart_ablation.png, models/model_rf_default.joblib")
