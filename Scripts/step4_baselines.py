# Step 4: train several classical models on the same features and compare them on the validation set
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, confusion_matrix

SEED = 42


def load(split):
    d = np.load(f"patches, features and scores (numpy arrays)/features_{split}.npz")
    # F4 = HOG + colour + LBP
    return np.hstack([d["hog"], d["color"], d["lbp"]]), d["y"]


Xtr, ytr = load("train")
Xva, yva = load("valid")

# scaler fitted on train only
sc = StandardScaler().fit(Xtr)
Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

# PCA copy for the slow models (KNN, RBF-SVM), also fitted on train only
pca = PCA(n_components=150, random_state=SEED).fit(Xtr)
Ptr, Pva = pca.transform(Xtr), pca.transform(Xva)
print("PCA keeps %.0f%% of the variance with 150 components" % (100 * pca.explained_variance_ratio_.sum()))

# (name, model, uses PCA features?)
models = [
    ("Logistic Regression", LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000), False),
    ("Linear SVM", LinearSVC(C=0.001, class_weight="balanced", max_iter=5000), False),
    ("RBF SVM", SVC(C=1, kernel="rbf", class_weight="balanced"), True),
    ("Random Forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=4, random_state=SEED), False),
    ("Extra Trees", ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=4, random_state=SEED), False),
    ("Gradient Boosting", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), True),
    ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED), False),
    ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=4), True),
    ("Naive Bayes", GaussianNB(), False),
]

rows = []
for name, model, use_pca in models:
    a, b = (Ptr, Pva) if use_pca else (Xtr, Xva)
    t = time.time()
    model.fit(a, ytr)
    pred = model.predict(b)
    tn, fp, fn, tp = confusion_matrix(yva, pred).ravel()
    rows.append(dict(model=name,
                     accuracy=accuracy_score(yva, pred),
                     recall=recall_score(yva, pred),
                     precision=precision_score(yva, pred, zero_division=0),
                     f1=f1_score(yva, pred),
                     f2=fbeta_score(yva, pred, beta=2),
                     TP=tp, FP=fp, FN=fn, TN=tn,
                     seconds=round(time.time() - t)))
    r = rows[-1]
    print(f"{name:20s} acc {r['accuracy']:.3f}  recall {r['recall']:.3f}  precision {r['precision']:.3f}  "
          f"F2 {r['f2']:.3f}  FN {fn}  ({r['seconds']}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("csvs/results_baselines.csv", index=False)

# one bar chart per metric
for metric in ["accuracy", "recall", "precision"]:
    d = df.sort_values(metric)
    plt.figure(figsize=(8, 5))
    plt.barh(d["model"], d[metric], color="steelblue")
    for i, v in enumerate(d[metric]):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center")
    plt.xlim(0, 1.1)
    plt.xlabel(metric.capitalize())
    plt.title(f"Validation {metric} (HOG + colour + LBP features)")
    plt.tight_layout()
    plt.savefig(f"charts/chart_{metric}.png", dpi=150)
    plt.close()
print("saved results_baselines.csv and chart_accuracy/recall/precision.png")
