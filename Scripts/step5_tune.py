# Step 5: tune the top 3 models with Optuna (objective = F2 score on the validation set)
import sys
import time
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, confusion_matrix

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load(split):
    d = np.load(f"patches, features and scores (numpy arrays)/features_{split}.npz")
    return np.hstack([d["hog"], d["color"], d["lbp"]]), d["y"]


Xtr, ytr = load("train")
Xva, yva = load("valid")
sc = StandardScaler().fit(Xtr)
Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
pca = PCA(n_components=150, random_state=SEED).fit(Xtr)
Ptr, Pva = pca.transform(Xtr), pca.transform(Xva)

# smaller random subset of train for the slow models while searching
rng = np.random.default_rng(SEED)
sub = rng.choice(len(ytr), 10000, replace=False)


def run(name, make_model, suggest, n_trials, a, b, subset):
    """Search on (optionally) a subset of train, then refit the best params on all of train."""
    a_s, y_s = (a[sub], ytr[sub]) if subset else (a, ytr)

    def objective(trial):
        model = make_model(suggest(trial))
        model.fit(a_s, y_s)
        return fbeta_score(yva, model.predict(b), beta=2)

    t = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f"sqlite:///optuna/optuna_{name}.db", study_name=name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    print(f"{name}: best search F2 {study.best_value:.4f} with {study.best_params} ({time.time()-t:.0f}s)", flush=True)

    model = make_model(study.best_params).fit(a, ytr)
    pred = model.predict(b)
    tn, fp, fn, tp = confusion_matrix(yva, pred).ravel()
    joblib.dump(model, f"models/model_{name}.joblib")
    res = dict(model=name, accuracy=accuracy_score(yva, pred), recall=recall_score(yva, pred),
               precision=precision_score(yva, pred), f2=fbeta_score(yva, pred, beta=2),
               TP=tp, FP=fp, FN=fn, TN=tn, params=str(study.best_params))
    print(f"  full refit -> acc {res['accuracy']:.3f} recall {res['recall']:.3f} "
          f"precision {res['precision']:.3f} F2 {res['f2']:.3f} FN {fn}", flush=True)
    return res


results = []

results.append(run(
    "linear_svm",
    lambda p: LinearSVC(C=p["C"], class_weight="balanced", max_iter=3000),
    lambda t: {"C": t.suggest_float("C", 1e-5, 1e-1, log=True)},
    15, Xtr, Xva, subset=False))

results.append(run(
    "rbf_svm",
    lambda p: SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight="balanced"),
    lambda t: {"C": t.suggest_float("C", 0.1, 100, log=True),
               "gamma": t.suggest_float("gamma", 1e-5, 1e-1, log=True)},
    25, Ptr, Pva, subset=True))

results.append(run(
    "random_forest",
    lambda p: RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                     class_weight="balanced_subsample", n_jobs=4, random_state=SEED),
    lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 300),
               "max_depth": t.suggest_int("max_depth", 10, 40),
               "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 5),
               "max_features": t.suggest_float("max_features", 0.005, 0.05, log=True)},
    12, Xtr, Xva, subset=True))

pd.DataFrame(results).to_csv("csvs/results_tuned.csv", index=False)
print("saved results_tuned.csv")
